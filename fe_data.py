#!/usr/bin/env python3.14

def sub_sub_database_features(sub_sub_database='', folder_output='', folder_output_renew=False, name_output=''):
    import pandas as pd
    from machine_learning.utilities import makedir

    # The output folder
    makedir(folder_output+'/', renew_folder = folder_output_renew)
    
    # Read database
    df = pd.read_csv(sub_sub_database)

    # Make database with transformed features
    df['Density_sum']            =    (df['Density_1']              + df['Density_2'])/ df['Density_Fe']
    df['Density_diff']           = abs(df['Density_1']              - df['Density_2'])/ df['Density_Fe']
    df['Density_aver']           =    (df['Density_1']              + df['Density_2'])/(2 * df['Density_Fe'])

    df['Eneg_sum']               =    (df['Eneg_1']              + df['Eneg_2'])/ df['Eneg_Fe']
    df['Eneg_diff']              = abs(df['Eneg_1']              - df['Eneg_2'])/ df['Eneg_Fe']
    df['Eneg_aver']              =    (df['Eneg_1']              + df['Eneg_2'])/(2 * df['Eneg_Fe'])

    df['Magnetic_sum']           =     df['Magnetic_1']          + df['Magnetic_2']
    df['Magnetic_diff']          = abs(df['Magnetic_1']          - df['Magnetic_2'])

    df['Melting_sum']            =    (df['Melting_1']           + df['Melting_2'])/ df['Melting_Fe']
    df['Melting_diff']           = abs(df['Melting_1']           - df['Melting_2'])/ df['Melting_Fe']
    df['Melting_aver']           =    (df['Melting_1']           + df['Melting_2'])/(2 * df['Melting_Fe'])

    df['Number_sum']             =    (df['Number_1']            + df['Number_2'])/ df['Number_Fe']
    df['Number_diff']            = abs(df['Number_1']            - df['Number_2'])/ df['Number_Fe']
    df['Number_aver']            =    (df['Number_1']            + df['Number_2'])/(2 * df['Number_Fe'])

    df['Radius_sum']             =    (df['Radius_1']            + df['Radius_2'])/df['Radius_Fe']
    df['Radius_diff']            = abs(df['Radius_1']            - df['Radius_2'])/df['Radius_Fe']
    df['Radius_aver']            =    (df['Radius_1']            + df['Radius_2'])/(2 * df['Radius_Fe'])

    try:
        df['Substitution_sum']       =     df['Substitution_1']      + df['Substitution_2']
        df['Substitution_diff']      = abs(df['Substitution_1']      - df['Substitution_2'])
        df_substitution              = df[['Substitution_1', 'Substitution_2']]
        df['Substitution_max']       = df_substitution.max(axis='columns')
    except KeyError:
        pass

    df['Vacancy_1_sum']          =     df['Vacancy_11']          + df['Vacancy_21']
    df['Vacancy_1_diff']         = abs(df['Vacancy_11']          - df['Vacancy_21'])

    df['Vacancy_2_sum']          =     df['Vacancy_12']          + df['Vacancy_22']
    df['Vacancy_2_diff']         = abs(df['Vacancy_12']          - df['Vacancy_22'])

    df['Vacancy_3_sum']          =     df['Vacancy_13']          + df['Vacancy_23']
    df['Vacancy_3_diff']         = abs(df['Vacancy_13']          - df['Vacancy_23'])

    df['Valence_sum']            =    (df['Valence_1']           + df['Valence_2'])/ df['Valence_Fe']
    df['Valence_diff']           = abs(df['Valence_1']           - df['Valence_2'])/ df['Valence_Fe']
    df['Valence_aver']           =    (df['Valence_1']           + df['Valence_2'])/(2 * df['Valence_Fe'])

    try:
        df['Stress_sum']             =     df['Stress_sub_1']        + df['Stress_sub_2']
        df['Stress_diff']            = abs(df['Stress_sub_1']        - df['Stress_sub_2'])
        df['Stress_binding']         =     df['Stress_sub_sub']      - (df['Stress_sub_1'] + df['Stress_sub_2'])
    except KeyError:
        pass

    # Writing database with transformed features
    df_features = df[['Elements',               # 1

                      'Density_sum',            # 2
                      'Density_diff',           # 3
                      'Density_aver',           # 4

                      'Eneg_sum',               # 5
                      'Eneg_diff',              # 6
                      'Eneg_aver',              # 7

                      'Magnetic_sum',           # 8
                      'Magnetic_diff',          # 9

                      'Melting_sum',            # 10
                      'Melting_diff',           # 11
                      'Melting_aver',           # 12

                      'Number_sum',             # 13
                      'Number_diff',            # 14
                      'Number_aver',            # 15

                      'Radius_sum',             # 16
                      'Radius_diff',            # 17
                      'Radius_aver',            # 18

                      'Vacancy_1_sum',          # 19
                      'Vacancy_1_diff',         # 20

                      'Vacancy_2_sum',          # 21
                      'Vacancy_2_diff',         # 22

                      'Vacancy_3_sum',          # 23
                      'Vacancy_3_diff',         # 24

                      'Valence_sum',            # 25
                      'Valence_diff',           # 26
                      'Valence_aver',           # 27

                      'Energy_binding']]        # 28
    df_features.to_csv(folder_output+'/'+name_output, index=False)