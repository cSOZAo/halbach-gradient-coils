import numpy as np
from typing import List
import logging

from .data_structures import CoilPart, DataStructure, FieldErrors, SolutionErrors, TargetField
from .process_raw_loops import biot_savart_calc_b
from .field_component import project_field

log = logging.getLogger(__name__)


def optimal_layout_current(target, layout) -> float:
    """Least-squares current magnitude for B_parallel, without pointwise ratios.

    A nonzero target with no generated field has no finite matching current
    and is reported as unavailable (NaN). Two zero fields need zero current.
    """
    target = np.asarray(target, dtype=float)
    layout = np.asarray(layout, dtype=float)
    if target.shape != layout.shape or target.size == 0:
        raise ValueError('Target and layout must contain matching field samples')
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(layout)):
        return np.nan
    scale = float(np.max(np.abs(layout)))
    if scale == 0:
        return 0.0 if np.all(target == 0) else np.nan
    normalized = layout / scale
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        current = (np.dot(normalized, target) / np.dot(normalized, normalized)) / scale
    return float(abs(current)) if np.isfinite(current) else np.nan


def _relative_field_errors(actual, reference):
    """Maximum/mean percentage; zero-reference comparisons have explicit limits."""
    denominator = np.max(np.abs(reference))
    residual = np.abs(actual - reference)
    if denominator == 0:
        return (0.0, 0.0) if np.all(residual == 0) else (np.nan, np.nan)
    relative = residual / denominator * 100
    return float(np.max(relative)), float(np.mean(relative))


def evaluate_field_errors(coil_parts: List[CoilPart], input_args: DataStructure, target_field: TargetField, sf_b_field: np.ndarray) -> (List[CoilPart], SolutionErrors):
    """
    Calculate relative errors between different input and output fields.

    Initialises the following properties of the CoilParts:
        - field_by_loops
        - field_by_layout

    Depends on the following properties of the CoilParts:
        - wire_path

    Depends on the following input_args:
        - skip_postprocessing

    Updates the following properties of a CoilPart:
        - None

    Args:
        coil_parts (List[CoilPart]): List of CoilPart structures.
        input_args (struct): Input parameters.
        target_field (TargetField): The target magnetic field.
        sf_b_field (np.ndarray): TBD

    Returns:
        solution_errors (SolutionErrors): A structure containing field error values and other information.
    """
    for coil_part in coil_parts:
        # Calculate the combined field of the unconnected contours
        coil_part.field_by_loops2 = np.zeros((3, target_field.b.shape[1]))  # (3,257)
        for loop in coil_part.contour_lines:
            loop_field = biot_savart_calc_b(loop.v, target_field)
            coil_part.field_by_loops2 += loop_field
        # # Removed by BugFix 4d4c632
        # # coil_part.field_by_loops2 *= coil_part.contour_step

        # Calculate the field of connected, final layouts
        if not input_args.skip_postprocessing:
            coil_part.field_by_layout = biot_savart_calc_b(coil_part.wire_path.v, target_field)
            # # Removed by BugFix 4d4c632
            # # coil_part.field_by_layout *= coil_part.contour_step  # scaled with the current of the discretization
        else:
            coil_part.field_by_layout = coil_part.field_by_loops2

    # # Added by BugFix 4d4c632
    # %Scale the SF field to 1A
    coil_part = coil_parts[0]  # Use the first coil part to get the contour step
    sf_b_field_1A = sf_b_field / coil_part.contour_step
    target_field_1A = target_field
    target_field_1A.b = target_field_1A.b / coil_part.contour_step

    """
    The provided code segment initializes the fields in coil_parts, calculates the field by loops and layouts,
    and finds the ideal current strength.
    """
    # End of part 1

    # Part 2: Find the current polarity for the different coil parts regarding the target field
    # Create all possible polarities as binary representations
    possible_polarities = [list(format(x, '0' + str(len(coil_parts)) + 'b')) for x in range(2 ** len(coil_parts))]

    # Convert binary representations to integers and set 0 to -1
    possible_polarities = [[1 if digit == '1' else -1 for digit in polarity] for polarity in possible_polarities]

    # Combine the coil_part fields scaled with all possible polarities and choose the one
    # which is closest to the target field
    combined_field_layout = np.zeros((len(possible_polarities), *coil_part.field_by_layout.shape))
    combined_field_loops = np.zeros((len(possible_polarities), *coil_part.field_by_loops2.shape))
    pol_projections_loops = np.zeros(len(possible_polarities))
    pol_projections_layout = np.zeros(len(possible_polarities))

    for pol_ind in range(len(possible_polarities)):
        for part_ind in range(len(coil_parts)):
            combined_field_layout[pol_ind] += possible_polarities[pol_ind][part_ind] * \
                coil_parts[part_ind].field_by_layout
            combined_field_loops[pol_ind] += possible_polarities[pol_ind][part_ind] * \
                coil_parts[part_ind].field_by_loops2

        # Project the combined field onto the target field
        pol_projections_layout[pol_ind] = np.linalg.norm(combined_field_layout[pol_ind] - target_field.b)  # Fails?!?
        pol_projections_loops[pol_ind] = np.linalg.norm(combined_field_loops[pol_ind] - target_field.b)  # Passes

        # # Added by BugFix 4d4c632
        pol_projections_layout[pol_ind] = np.linalg.norm(project_field(
            combined_field_layout[pol_ind] - sf_b_field_1A.T, input_args))
        pol_projections_loops[pol_ind] = np.linalg.norm(project_field(
            combined_field_loops[pol_ind] - sf_b_field_1A.T, input_args))
    """
    This code segment covers the second part of your MATLAB code, including generating possible polarities,
    combining fields with different polarities, and projecting the combined fields onto the target field. 
    """
    # End of part 2
    # Part 3: Choose the best combination and adjust the current direction

    # Find the best direction for layout and loops
    best_dir_layout = np.argmin(pol_projections_layout)
    best_dir_loops = np.argmin(pol_projections_loops)

    # Choose the best combination for layout and loops
    combined_field_layout = combined_field_layout[best_dir_layout]  # Fail
    combined_field_loops = combined_field_loops[best_dir_loops]  # Pass

    # Adjust the current direction for the layout (in case of the wrong direction)
    if not input_args.skip_postprocessing:
        for part_ind in range(len(coil_parts)):
            if possible_polarities[best_dir_layout][part_ind] != 1:
                coil_parts[part_ind].wire_path.v = np.fliplr(coil_parts[part_ind].wire_path.v)
                coil_parts[part_ind].wire_path.uv = np.fliplr(coil_parts[part_ind].wire_path.uv)

    # Adjust the current direction for the loops (in case of the wrong direction)
    for part_ind in range(len(coil_parts)):
        if possible_polarities[best_dir_loops][part_ind] != 1:
            for loop_ind in range(len(coil_parts[part_ind].contour_lines)):
                coil_parts[part_ind].contour_lines[loop_ind].v = np.fliplr(
                    coil_parts[part_ind].contour_lines[loop_ind].v)
                coil_parts[part_ind].contour_lines[loop_ind].uv = np.fliplr(
                    coil_parts[part_ind].contour_lines[loop_ind].uv)

    """
    This code segment covers the third part of your MATLAB code, including choosing the best combination, adjusting
    the current direction for layout and loops if necessary.
    """
    # End of part 3

    # Part 4: Calculate field errors and return results
    # Extract z-components of the fields

    # # Modified by BugFix 4d4c632
    # # target_z = target_field.b[2, :]  # Tranpose into MATLAB shape (3,n)
    # # sf_z = sf_b_field[:, 2]  # Field of stream function (Transposed, because it is Python shaped (n,3))
    target_z = project_field(target_field_1A.b, input_args)
    sf_z = project_field(sf_b_field_1A.T, input_args)

    layout_z = project_field(combined_field_layout, input_args)
    loop_z = project_field(combined_field_loops, input_args)

    # Calculate relative errors
    field_error_vals = FieldErrors()
    (field_error_vals.max_rel_error_layout_vs_target,
     field_error_vals.mean_rel_error_layout_vs_target) = _relative_field_errors(layout_z, target_z)
    (field_error_vals.max_rel_error_unconnected_contours_vs_target,
     field_error_vals.mean_rel_error_unconnected_contours_vs_target) = _relative_field_errors(loop_z, target_z)
    (field_error_vals.max_rel_error_layout_vs_stream_function_field,
     field_error_vals.mean_rel_error_layout_vs_stream_function_field) = _relative_field_errors(layout_z, sf_z)
    (field_error_vals.max_rel_error_unconnected_contours_vs_stream_function_field,
     field_error_vals.mean_rel_error_unconnected_contours_vs_stream_function_field) = _relative_field_errors(loop_z, sf_z)

    # Go back to the fields for 1 Ampere (Unit Current)
    combined_field_layout_per1Amp = np.zeros_like(combined_field_layout)
    combined_field_loops_per1Amp = np.zeros_like(combined_field_layout)

    for part_ind in range(len(coil_parts)):
        coil_part = coil_parts[part_ind]
        # # Modified by BugFix 4d4c632
        # # combined_field_layout_per1Amp += (coil_parts[part_ind].field_by_layout /
        # #                                   np.max([coil_parts[x].contour_step for x in range(len(coil_parts))])) * possible_polarities[best_dir_layout][part_ind]
        combined_field_layout_per1Amp += coil_part.field_by_layout * possible_polarities[best_dir_layout][part_ind]
        # # # Passes
        # # combined_field_loops_per1Amp += (coil_parts[part_ind].field_by_loops2 /
        # #                                  np.max([coil_parts[x].contour_step for x in range(len(coil_parts))])) * possible_polarities[best_dir_loops][part_ind]

        #   File "C:\Dev\CoilGen-Python\pyCoilGen\pyCoilGen_release.py", line 402, in pyCoilGen
        #     raise e
        #   File "C:\Dev\CoilGen-Python\pyCoilGen\pyCoilGen_release.py", line 369, in pyCoilGen
        #     coil_parts, solution_errors = evaluate_field_errors(
        #   File "C:\Dev\CoilGen-Python\pyCoilGen\sub_functions\evaluate_field_errors.py", line 191, in evaluate_field_errors
        #     combined_field_loops_per1Amp += coil_parts[part_ind].field_by_loops * \
        # ValueError: operands could not be broadcast together with shapes (3,3561) (3,3561,56) (3,3561)

        combined_field_loops_per1Amp += coil_part.field_by_loops2 * possible_polarities[best_dir_loops][part_ind]

    # Calculate the ideal current strength for the connected layout to match the target field
    opt_current_layout = optimal_layout_current(target_z, layout_z)

    # Return results
    solution_errors = SolutionErrors(field_error_vals=field_error_vals)
    solution_errors.combined_field_layout = combined_field_layout
    solution_errors.combined_field_loops = combined_field_loops
    solution_errors.combined_field_layout_per1Amp = combined_field_layout_per1Amp
    solution_errors.combined_field_loops_per1Amp = combined_field_loops_per1Amp
    solution_errors.opt_current_layout = opt_current_layout
    # # Added by BugFix 4d4c632
    solution_errors.sf_b_field_1A = sf_b_field_1A
    solution_errors.target_field_1A = target_field_1A

    return coil_parts, solution_errors
