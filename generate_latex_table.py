import pandas as pd

def to_latex_table_metrics(df):
    """
    Converts a pandas DataFrame to the first custom LaTeX table format for metrics.
    """
    
    # Select and rename columns for the table
    table_df = df[['error_noisy', 'error_ekf', 'mae_noisy', 'mae_ekf', 'rmse_noisy', 'rmse_ekf', 'std_noisy', 'std_ekf']].copy()
    
    # Calculate averages
    averages = table_df.mean()
    
    # Start LaTeX table string
    latex_string = "\\begin{table}[H]\n"
    latex_string += "\\centering\n"
    latex_string += "\\caption{Resultados das iterações para trajetória simulada e EKF}\n"
    latex_string += "\\label{tab:metricas}\n"
    latex_string += "\\begin{tabular}{ccccccccc}\n"
    latex_string += "\\toprule\n"
    latex_string += "Iteração & \\multicolumn{2}{c}{Erro último ponto (m)} & \\multicolumn{2}{c}{MAE (m)} & \\multicolumn{2}{c}{RMSE (m)} & \\multicolumn{2}{c}{Desvio padrão (m)} \\\\\n"
    latex_string += "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9}\n"
    latex_string += " & Simulada & EKF & Simulada & EKF & Simulada & EKF & Simulada & EKF \\\\\n"
    latex_string += "\\midrule\n"
    
    # Add data rows
    for i, row in table_df.iterrows():
        row_values = [f"{i+1}"] + [f"{val:.4f}" for val in row]
        latex_string += " & ".join(row_values) + " \\\\\n"
        
    latex_string += "\\midrule\n"
    
    # Add averages row
    avg_values = ["Média"] + [f"{val:.4f}" for val in averages]
    latex_string += " & ".join(avg_values) + " \\\\\n"
    
    latex_string += "\\bottomrule\n"
    latex_string += "\\end{tabular}\n"
    latex_string += "\\end{table}\n"
    
    return latex_string

def to_latex_table_performance(df):
    """
    Converts a pandas DataFrame to the second custom LaTeX table format for performance.
    """
    
    # Calculate relative performance
    df['desempenho_relativo'] = (1 - df['mae_ekf'] / df['mae_noisy']) * 100
    
    # Select columns for the table
    table_df = df[['desempenho_relativo', 'distance_gt', 'distance_noisy', 'distance_ekf']].copy()
    
    # Calculate averages
    averages = table_df.mean()
    
    # Start LaTeX table string
    latex_string = "\n\\begin{table}[H]\n"
    latex_string += "\\centering\n"
    latex_string += "\\caption{Desempenho relativo e distâncias percorridas}\n"
    latex_string += "\\label{tab:desempenho}\n"
    latex_string += "\\begin{tabular}{ccccc}\n"
    latex_string += "\\toprule\n"
    latex_string += "Iteração & Desempenho relativo EKF (\\%) & \\multicolumn{3}{c}{Distância total percorrida (m)} \\\\\n"
    latex_string += "\\cmidrule(lr){3-5}\n"
    latex_string += " & & GT & Simulada & EKF \\\\\n"
    latex_string += "\\midrule\n"
    
    # Add data rows
    for i, row in table_df.iterrows():
        row_values = [f"{i+1}", f"{row['desempenho_relativo']:.2f}", f"{row['distance_gt']:.2f}", f"{row['distance_noisy']:.2f}", f"{row['distance_ekf']:.2f}"]
        latex_string += " & ".join(row_values) + " \\\\\n"
        
    latex_string += "\\midrule\n"
    
    # Add averages row
    avg_values = ["Média", f"{averages['desempenho_relativo']:.2f}", f"{averages['distance_gt']:.2f}", f"{averages['distance_noisy']:.2f}", f"{averages['distance_ekf']:.2f}"]
    latex_string += " & ".join(avg_values) + " \\\\\n"
    
    latex_string += "\\bottomrule\n"
    latex_string += "\\end{tabular}\n"
    latex_string += "\\end{table}\n"
    
    return latex_string

def main():
    """
    Main function to read data, calculate averages, and generate LaTeX tables.
    """
    # Load the dataset
    df = pd.read_csv('odometry_data.csv')
    
    # Generate the LaTeX tables
    latex_output_metrics = to_latex_table_metrics(df)
    latex_output_performance = to_latex_table_performance(df)
    
    # Combine and save the output to a file
    with open('odometry_table.tex', 'w') as f:
        f.write(latex_output_metrics)
        f.write(latex_output_performance)
        
    print("LaTeX tables have been saved to odometry_table.tex")

if __name__ == "__main__":
    main()
