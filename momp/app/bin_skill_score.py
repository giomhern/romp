from dataclasses import asdict
from itertools import product

from momp.lib.control import iter_list, make_case
from momp.lib.convention import Case
#from momp.lib.loader import cfg, setting
from momp.lib.loader import get_cfg, get_setting
#from momp.io.output import set_nested
from momp.utils.printing import tuple_to_str
from momp.io.dict import select_key_at_level
from momp.lib.control import filter_bins_in_window
from collections import defaultdict


#def bin_skill_score(BSS, RPS, AUC, skill_score, ref_model, ref_model_dir,
#                         years, years_clim, model, model_forecast_dir, obs_dir, thres_file
#                         members, max_forecast_day, day_bins, date_filter_year,
#                         file_pattern, mok, save_csv_score, plot_heatmap, **kwargs):

cfg, setting = get_cfg(), get_setting()

def skill_score_in_bins(cfg=cfg, setting=setting):

    # only execute for ensemble forecasts
    #if not cfg.get('probabilistic'):
    #if not getattr(cfg, "probabilistic", False):
    if not cfg.probabilistic:
        return

    from momp.io.output import save_score_results
    from momp.metrics.skill import create_score_results

    #result_overall = {}
    #result_binned = {}
    result_overall = defaultdict(dict)
    result_binned = defaultdict(dict)

    layout_pool = iter_list(vars(cfg))

    for combi in product(*layout_pool):
        case = make_case(Case, combi, vars(cfg))

        print(f"{'='*50}")
        print(f"processing {case.model} onset evaluation for verification window "
                f"{case.verification_window}, case: {case.case_name}")
        #print(f"processing bin skill score for {case.case_name}")

        day_bins_filtered = filter_bins_in_window(case.day_bins, case.verification_window)
        case.day_bins = day_bins_filtered

        case_cfg = {**asdict(setting), **asdict(case)}
#        print("\n\n\n members = ", case_cfg['members'])
#        print("\n\n\n max_forecast_day = ", case_cfg['max_forecast_day'])
#        print("\n\n\n cfg.max_forecast_day = ", cfg.max_forecast_day)
#        print("\n\n\n case.max_forecast_day = ", case.max_forecast_day)

#        from pprint import pprint
#        pprint(case_cfg)

        import time
        start = time.perf_counter()

        # Create bin skill score metrics
        score_results = create_score_results(**case_cfg)

        end = time.perf_counter()
        print(f"\n\n\n skill score Execution time: {end - start:.4f} seconds\n\n\n")
        
        # save score results as csv file
        binned_data, overall_scores = None, None
        if case_cfg['save_csv_score']:
            #save_score_results(score_results, **case_cfg)
            binned_data, overall_scores = save_score_results(score_results, **case_cfg)

        window_str = tuple_to_str(case.verification_window)
        result_binned[case.model][window_str] = binned_data
        result_overall[case.model][window_str] = overall_scores
        

        # heatmap plot
        if case_cfg['plot_heatmap_bss_auc']:
            try:
                from momp.graphics.heatmap import create_heatmap

                create_heatmap(score_results, **case_cfg)
            except ModuleNotFoundError as exc:
                print(f"Skipping heatmap plot because an optional graphics dependency is missing: {exc}")

        # reliability plot
        if case_cfg['plot_reliability']:
            try:
                from momp.graphics.reliability import plot_reliability_diagram

                plot_reliability_diagram(score_results["forecast_obs_df"], **case_cfg)
            except ModuleNotFoundError as exc:
                print(f"Skipping reliability plot because an optional graphics dependency is missing: {exc}")

#    print("\n score_results \n ", score_results['skill_results']['bin_fair_brier_skill_scores'])
#    print("\n binned_data \n", binned_data['Fair_Brier_Skill_Score'])


    max_forecast_day = cfg.max_forecast_day

    if 2 > 3:
        import pickle
        import os
        fout = os.path.join(cfg.dir_out,f"combi_binned_skill_scores_{max_forecast_day}day.pkl")
        with open(fout, "wb") as f:
            pickle.dump(result_binned, f)

        fout = os.path.join(cfg.dir_out,f"combi_overall_skill_scores_{max_forecast_day}day.pkl")
        with open(fout, "wb") as f:
            pickle.dump(result_overall, f)

    from pprint import pprint

    # panel heatmap plot for binned BSS and AUC
    if case_cfg['plot_panel_heatmap_skill']:
        #panel_portrait_bss_auc(result_binned, **case_cfg)
        for verification_window in cfg.verification_window_list:
            window_str = tuple_to_str(verification_window)
            # extract dict for specific window only
            result_binned_window = select_key_at_level(result_binned, 2, window_str)
            print("\n\n\n result_binned_window = ", result_binned_window)
            pprint(result_binned_window)
            try:
                from momp.graphics.panel_portrait_skill import panel_portrait_bss_auc

                panel_portrait_bss_auc(result_binned_window, verification_window, **vars(cfg))
            except ModuleNotFoundError as exc:
                print(f"Skipping panel heatmap plot because an optional graphics dependency is missing: {exc}")

    # bar plot for BSS, RPSS, AUC in window
    if case_cfg['plot_bar_bss_rpss_auc']:
        #panel_bar_bss_rpss_auc(result_overall, **case_cfg)
        #panel_bar_bss_rpss_auc(result_overall, **vars(cfg))
        for verification_window in cfg.verification_window_list:
            #if verification_window[0] != 1: # RPSS requires integration from 1 to end of window
            #    continue
            window_str = tuple_to_str(verification_window)
            result_overall_window = select_key_at_level(result_overall, 2, window_str)
            print("\n\n\n result_overall_window = ", result_overall_window)
            #pprint(result_overall_window)
            try:
                from momp.graphics.panel_bar_skill import panel_bar_bss_rpss_auc

                panel_bar_bss_rpss_auc(result_overall_window, verification_window, **vars(cfg))
            except ModuleNotFoundError as exc:
                print(f"Skipping bar plot because an optional graphics dependency is missing: {exc}")

#    # make spatial metrics plot --note it works only for whole country region, not subregion
#    if case_cfg['plot_spatial_far_mr_mae']:
#        ens_spatial_far_mr_mae_map()

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    skill_score_in_bins()
