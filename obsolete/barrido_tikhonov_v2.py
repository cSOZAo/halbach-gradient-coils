import os
import pandas as pd
import numpy as np

# Importamos la función desde el archivo que acabo de crear para ti
from gradiente_belen_santi_func import generar_gradiente

def realizar_barrido_inteligente():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_sweep_dir = os.path.join(base_dir, "resultados_barrido_tikhonov_v2")
    os.makedirs(output_sweep_dir, exist_ok=True)
    
    # 1. Distribución inicial de Tikhonov (barrido grueso entre 0 y 100k)
    # Nota: No usamos 0 exacto porque pyCoilGen se inestabiliza si Tikhonov es nulo, 
    # por lo que partimos con un valor muy bajo (1) pero efectivo.
    # El crecimiento es exponencial/logarítmico para abarcar todo el rango correctamente.
    tikhonov_coarse = [1, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
    
    # Parámetros de construcción fijos
    niveles = 26
    largo_cilindro = 0.44
    ejes_radios = {
        'x': 0.160,
        'y': 0.152,
        'z': 0.144
    }
    
    for eje, radio in ejes_radios.items():
        print(f"\n{'='*60}")
        print(f"BARRIDO GRUESO - Eje {eje.upper()} (Radio: {radio} m)")
        print(f"{'='*60}")
        
        axis_dir = os.path.join(output_sweep_dir, f"Eje_{eje.upper()}")
        os.makedirs(axis_dir, exist_ok=True)
        
        resultados_gruesos = []
        
        # =========================================================================
        # FASE 1: BARRIDO GRUESO
        # =========================================================================
        for tk in tikhonov_coarse:
            print(f"\n  -> [Grueso] Evaluando Tikhonov = {tk} ...")
            tk_dir = os.path.join(axis_dir, f"Tk_grueso_{tk}")
            os.makedirs(tk_dir, exist_ok=True)
            
            # Llamamos a nuestra nueva función
            metrics = generar_gradiente(
                gradient_axis=eje,
                cyl_height=largo_cilindro,
                cyl_radius=radio,
                tikhonov_factor=tk,
                num_levels=niveles,
                base_output_dir=tk_dir,
                show_plots=False  # Para no pausar el código
            )
            
            resultados_gruesos.append({
                'Fase': 'Grueso',
                'Tikhonov': tk,
                'Pendiente (mT/m*A)': metrics['slope_mTmA'],
                'Error Medio (%)': metrics['mean_rel_err_layout_pct']
            })
            
        df_grueso = pd.DataFrame(resultados_gruesos)
        
        # =========================================================================
        # FASE 2: BARRIDO FINO
        # =========================================================================
        # Asumimos que la "mejor" pendiente es la que tiene mayor eficiencia absoluta (mT/m*A)
        idx_mejor = df_grueso['Pendiente (mT/m*A)'].abs().idxmax()
        mejor_tk_grueso = df_grueso.loc[idx_mejor, 'Tikhonov']
        
        # Definir límites: el valor anterior y el valor siguiente de la lista gruesa
        idx_lista = tikhonov_coarse.index(mejor_tk_grueso)
        
        if idx_lista > 0:
            tk_prev = tikhonov_coarse[idx_lista - 1]
        else:
            tk_prev = 0.1  # Límite inferior si el mejor fue el primero

        if idx_lista < len(tikhonov_coarse) - 1:
            tk_next = tikhonov_coarse[idx_lista + 1]
        else:
            tk_next = mejor_tk_grueso * 2
            
        print(f"\n{'='*60}")
        print(f"[*] Mejor Tikhonov grueso para {eje.upper()} fue {mejor_tk_grueso}.")
        print(f"[*] Iniciando FASE 2: BARRIDO FINO entre {tk_prev} y {tk_next}...")
        print(f"{'='*60}")
        
        # Generar puntos uniformemente espaciados entre el anterior y el siguiente
        # Generamos 7 puntos, de los cuales filtraremos los que ya hemos medido (los bordes)
        tikhonov_fine_raw = np.linspace(tk_prev, tk_next, num=7)
        tikhonov_fine = [round(val, 2) for val in tikhonov_fine_raw if round(val, 2) not in tikhonov_coarse]
        
        resultados_finos = []
        for tk in tikhonov_fine:
            print(f"\n  -> [Fino] Evaluando Tikhonov = {tk} ...")
            tk_dir = os.path.join(axis_dir, f"Tk_fino_{tk}")
            os.makedirs(tk_dir, exist_ok=True)
            
            metrics = generar_gradiente(
                gradient_axis=eje,
                cyl_height=largo_cilindro,
                cyl_radius=radio,
                tikhonov_factor=tk,
                num_levels=niveles,
                base_output_dir=tk_dir,
                show_plots=False
            )
            
            resultados_finos.append({
                'Fase': 'Fino',
                'Tikhonov': tk,
                'Pendiente (mT/m*A)': metrics['slope_mTmA'],
                'Error Medio (%)': metrics['mean_rel_err_layout_pct']
            })
            
        # =========================================================================
        # CONSOLIDAR RESULTADOS
        # =========================================================================
        resultados_totales = resultados_gruesos + resultados_finos
        df_total = pd.DataFrame(resultados_totales)
        
        # Ordenar por el valor de Tikhonov
        df_total = df_total.sort_values(by='Tikhonov').reset_index(drop=True)
        
        # Identificar los mejores valores finales
        idx_mejor_final = df_total['Pendiente (mT/m*A)'].abs().idxmax()
        mejor_final = df_total.loc[idx_mejor_final]
        
        idx_menor_error = df_total['Error Medio (%)'].idxmin()
        menor_error = df_total.loc[idx_menor_error]
        
        # Guardar Reportes
        resumen_csv = os.path.join(axis_dir, f"Resumen_Completo_Eje_{eje.upper()}.csv")
        df_total.to_csv(resumen_csv, index=False)
        
        resumen_txt = os.path.join(axis_dir, f"Resumen_Completo_Eje_{eje.upper()}.txt")
        with open(resumen_txt, 'w', encoding='utf-8') as f:
            f.write(f"Resumen de Barrido Tikhonov Inteligente - EJE {eje.upper()}\n")
            f.write(f"Construcción: Radio={radio}m, Largo={largo_cilindro}m, Niveles={niveles}\n")
            f.write("-" * 70 + "\n")
            f.write(df_total.to_string(index=False) + "\n\n")
            f.write("-" * 70 + "\n")
            f.write(f"[*] MEJOR PENDIENTE ABSOLUTA FINAL:\n    Tikhonov = {mejor_final['Tikhonov']} (Fase: {mejor_final['Fase']})\n    Valor = {mejor_final['Pendiente (mT/m*A)']} mT/m*A\n\n")
            f.write(f"[*] MENOR ERROR MEDIO DEL CAMPO FINAL:\n    Tikhonov = {menor_error['Tikhonov']} (Fase: {menor_error['Fase']})\n    Valor = {menor_error['Error Medio (%)']} %\n")
                
        print(f"\nResumen final para el Eje {eje.upper()} guardado en: '{axis_dir}'")

if __name__ == '__main__':
    realizar_barrido_inteligente()
