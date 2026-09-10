import pandas as pd
import numpy as np

def make_training_features_outputs(training_df, ordered_nucs):


    out_arr_list = []
    feature_arr_list = []
    ordered_nucs = ["h:1", "h:2", "he:3", "he:4", "li:6", "li:7", "be:9", "b:10", "b:11", "c:12", "c:13", "n:14", "n:15", "o:16", "o:17", "o:18", "f:19", "ne:20", "ne:21", "ne:22", "na:23", "mg:24", "mg:25", "mg:26", "al:27", "si:28", "si:29", "si:30", "p:31", "s:32", "s:33", "s:34", "s:36", "cl:35", "cl:37", "ar:36", "ar:38", "ar:40", "k:39", "k:41", "ca:40", "ca:42", "ca:43", "ca:44", "ca:46", "ca:48", "sc:45", "ti:46", "ti:47", "ti:48", "ti:49", "ti:50", "v:50", "v:51", "cr:50", "cr:52", "cr:53", "cr:54", "mn:55", "fe:54", "fe:56", "fe:57", "fe:58", "co:59", "ni:58", "ni:60", "ni:61", "ni:62", "ni:64", "cu:63", "cu:65", "zn:64", "zn:66", "zn:67", "zn:68", "zn:70", "ga:69", "ga:71", "ge:70", "ge:72", "ge:73", "ge:74", "ge:76", "as:75", "se:74", "se:76", "se:77", "se:78", "se:80", "se:82", "br:79", "br:81", "kr:78", "kr:80", "kr:82", "kr:83", "kr:84", "kr:86", "rb:85", "rb:87", "sr:84", "sr:86", "sr:87", "sr:88", "y:89", "zr:90", "zr:91", "zr:92", "zr:94", "zr:96", "nb:93", "mo:92", "mo:94", "mo:95", "mo:96", "mo:97", "mo:98", "mo:100", "ru:96", "ru:98", "ru:99", "ru:100", "ru:101", "ru:102", "ru:104", "rh:103", "pd:102", "pd:104", "pd:105", "pd:106", "pd:108", "pd:110", "ag:107", "ag:109", "cd:106", "cd:108", "cd:110", "cd:111", "cd:112", "cd:113", "cd:114", "cd:116", "in:113", "in:115", "sn:112", "sn:114", "sn:115", "sn:116", "sn:117", "sn:118", "sn:119", "sn:120", "sn:122", "sn:124", "sb:121", "sb:123", "te:120", "te:122", "te:123", "te:124", "te:125", "te:126", "te:128", "te:130", "i:127", "xe:124", "xe:126", "xe:128", "xe:129", "xe:130", "xe:131", "xe:132", "xe:134", "xe:136", "cs:133", "ba:130", "ba:132", "ba:134", "ba:135", "ba:136", "ba:137", "ba:138", "la:138", "la:139", "ce:136", "ce:138", "ce:140", "ce:142", "pr:141", "nd:142", "nd:143", "nd:144", "nd:145", "nd:146", "nd:148", "nd:150", "sm:144", "sm:147", "sm:148", "sm:149", "sm:150", "sm:152", "sm:154", "eu:151", "eu:153", "gd:152", "gd:154", "gd:155", "gd:156", "gd:157", "gd:158", "gd:160", "tb:159", "dy:156", "dy:158", "dy:160", "dy:161", "dy:162", "dy:163", "dy:164", "ho:165", "er:162", "er:164", "er:166", "er:167", "er:168", "er:170", "tm:169", "yb:168", "yb:170", "yb:171", "yb:172", "yb:173", "yb:174", "yb:176", "lu:175", "lu:176", "hf:174", "hf:176", "hf:177", "hf:178", "hf:179", "hf:180", "ta:181", "w:180", "w:182", "w:183", "w:184", "w:186", "re:185", "re:187", "os:184", "os:186", "os:187", "os:188", "os:189", "os:190", "os:192", "ir:191", "ir:193", "pt:190", "pt:192", "pt:194", "pt:195", "pt:196", "pt:198", "au:197", "hg:196", "hg:198", "hg:199", "hg:200", "hg:201", "hg:202", "hg:204", "tl:203", "tl:205", "pb:204", "pb:206", "pb:207", "pb:208", "bi:209", "th:232", "pa:231", "u:234", "u:235", "u:238"]

    for run_lbl in training_df['run_lbl'].unique():
        parent_child_nuc_arr = np.zeros((len(ordered_nucs), len(ordered_nucs)))
        reduced_df = training_df.loc[training_df['run_lbl'] == run_lbl]
        num_dens_idx = reduced_df.columns.get_loc("num_dens_(atoms/cm3)")
        for df_row in reduced_df.itertuples(index=False):
            child_nuc = df_row.nuclide
            parent_nuc = df_row.block_name.replace(":", "-")
            parent_child_nuc_arr[ordered_nucs.index(parent_nuc), ordered_nucs.index(child_nuc)] = df_row[num_dens_idx]
            t_irr = df_row.t_irr
            avg_flux_mag = df_row.avg_flux_mag
            flux_spec_shape = eval(df_row.flux_spec_shape)
        #Each run_lbl is associated with a single combination of each of the features    
        feature_arr_list.append(np.array((t_irr, avg_flux_mag, *flux_spec_shape)))
        out_arr_list.append(parent_child_nuc_arr)
    return out_arr_list, feature_arr_list    