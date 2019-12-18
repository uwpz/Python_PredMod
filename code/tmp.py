


np.array(["full_or_part_time_employment_stat"])


cols = np.array(["full_or_part_time_employment_stat",'country_of_birth_self',"target","encode_flag"])

tmp = TargetEncoding(features = ["full_or_part_time_employment_stat",'country_of_birth_self'], encode_flag_column = "encode_flag",
                     target = "target").fit_transform(df[cols])