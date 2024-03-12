import pandas as pd
from multiprocessing import Process, Pool
import pm4py
from pm4py.algo.conformance.alignments.petri_net import algorithm
from pm4py.algo.conformance.alignments.petri_net import variants


# def execute(df, miner="inductive", variance=3):
def align(df, name_df, miner="alpha", noise_threshold=0.2):
    print('Test ' + name_df + " " + miner + " " + str(noise_threshold))
    # discover petri net
    print(name_df)
    # if name_df == "prFm6":
    #   pn, im, fm = pm4py.read_pnml("pm4py/data/prFm6.pnml")

    #  print(im, fm)
    # name_df = "prFm6"

    # elif name_df == "prGm6":
    #   pn, im, fm = pm4py.read_pnml("pm4py/data/prGm6.pnml")
    # name_df = "prGm6"
    #  print(im, fm)
    if miner == "alpha":
        # alpha miner
        pn, im, fm = pm4py.discover_petri_net_alpha(df)
    else:
        # inductive miner
        pn, im, fm = pm4py.discover_petri_net_inductive(df, noise_threshold)
    print("Model discovered")

    variance = 3
    #miner = "no"

    pm4py.save_vis_petri_net(pn, im, fm, file_path="visualization/petri_net/" + name_df + "_" + miner + "_" + str(
        variance) + ".png")

    print("Model saved")
    result = algorithm.create_data(df, pn, im, fm, variance, name_df, miner, variant=variants.a_star)
    result.to_pickle("results/" + name_df + "_" + miner + "_" + str(variance) + ".pkl")

    print(result)


def get_var_name(var):
    for name, value in globals().items():
        if value is var:
            return name


# df_italian = pm4py.read_xes("pm4py/data/ItalianHelpdeskFinal_complete.xes")
# bpi12 = pm4py.read_xes("pm4py/data/BPI_Challenge_2012.xes")
# sepsis = pm4py.read_xes("pm4py/data/Sepsis Cases - Event Log.xes")
# road = pm4py.read_xes("pm4py/data/Road_Traffic_Fine_Management_Process.xes")

def test_process():
    procs = []

    for d in data:
        get_var_name()
        proc = Process(target=align, args=[d, ])
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


if __name__ == "__main__":
    #prFm6 = pm4py.read_xes("pm4py/data/prFm6.xes")
    #prGm6 = pm4py.read_xes("pm4py/data/prGm6.xes")

    #df_problems = pm4py.format_dataframe(pd.read_csv('pm4py/data/running_example_broken.csv', sep=';'),
     #                                    case_id='case:concept:name', activity_key='concept:name',
      #                                   timestamp_key='time:timestamp')

    #permit = pm4py.read_xes("pm4py/data/PermitLog.xes")
    #prepaid = pm4py.read_xes("pm4py/data/PrepaidTravelCost.xes")
    #international_declaration = pm4py.read_xes("pm4py/data/InternationalDeclarations.xes")
    #request = pm4py.read_xes("pm4py/data/RequestForPayment.xes")

    domestic = pm4py.read_xes("pm4py/data/DomesticDeclarations.xes")

    #data = [prepaid, permit]

    #data = [df_problems]

    #data = [prFm6, prGm6]
    #data = [request, international_declaration]

    data = [domestic]
    procs = []

    for d in data:
        name_df = get_var_name(d)
        proc_alpha = Process(target=align, args=[d, name_df])
        proc_im = Process(target=align, args=[d, name_df, "inductive"])

        procs.append(proc_alpha)
        procs.append(proc_im)

        proc_alpha.start()
        proc_im.start()

    for proc in procs:
        proc.join()
    
    print("Done")

    #for d in data:
     #   name_df = get_var_name(d)
      #  align(d, name_df)

# inductive miner
# pn, im, fm = pm4py.discover_petri_net_inductive(df)
# alpha miner
# alpha_pn, alpha_im, alpha_fm = pm4py.discover_petri_net_alpha(df)
# heuristic miner, param: dependency_threshold
# heuristic_pn, heuristic_im, heuristic_fm = pm4py.discover_petri_net_heuristics(df, dependency_threshold=0.99)

# italian df:
# alpha_pn_it, alpha_im_it, alpha_fm_it = pm4py.discover_petri_net_alpha(df_italian)
# heuristic_pn_it, heuristic_im_it, heuristic_fm_it = pm4py.discover_petri_net_heuristics(df_italian, dependency_threshold=0.99)
# inductive_pn_it_noise, inductive_im_it_noise, inductive_fm_it_noise = pm4py.discover_petri_net_inductive(df_italian, noise_threshold=0.2)
# inductive_pn_it, inductive_im_it, inductive_fm_it = pm4py.discover_petri_net_inductive(df_italian)

# Sepsis
# pn_alpha_sepsis, im_alpha_sepsis, fm_alpha_sepsis =  pm4py.discover_petri_net_alpha(df_sepsis)
# pn_im_sepsis, im_im_sepsis, fm_im_sepsis = pm4py.discover_petri_net_inductive(df_sepsis)

# pm4py.view_petri_net(pn_alpha_sepsis, im_alpha_sepsis, fm_alpha_sepsis)
# pm4py.view_petri_net(pn_im_sepsis, im_im_sepsis, fm_im_sepsis)

# print(algorithm.create_data(df_sepsis, pn_im_sepsis, im_im_sepsis, fm_im_sepsis, variant=variants.a_star))


# pm4py.view_petri_net(pn_bpi, im_bpi, fm_bpi)
# print(algorithm.create_data(df_bpi12, pn_bpi, im_bpi, fm_bpi, variant=variants.a_star))


# pm4py.view_petri_net(alpha_pn_it, alpha_im_it, alpha_fm_it)
# pm4py.view_petri_net(heuristic_pn_it, heuristic_im_it, heuristic_fm_it)
# pm4py.view_petri_net(inductive_pn_it, inductive_im_it, inductive_fm_it)
# pm4py.view_petri_net(inductive_pn_it_noise, inductive_im_it_noise, inductive_fm_it_noise)
# pm4py.view_petri_net(pn, im, fm)

# pm4py.view_petri_net(pn, im, fm)


# log = pm4py.convert_to_event_log(df_problems)

# pn_list = []
# im_list = []
# fm_list = []

# pn_list.append(alpha_pn_it)
# pn_list.append(heuristic_pn_it)
# pn_list.append(inductive_pn_it)
# pn_list.append(inductive_pn_it_noise)

# im_list.append(alpha_im_it)
# im_list.append(heuristic_im_it)
# im_list.append(inductive_im_it)
# im_list.append(inductive_im_it_noise)

# fm_list.append(alpha_fm_it)
# fm_list.append(heuristic_fm_it)
# fm_list.append(inductive_fm_it)
# fm_list.append(inductive_fm_it_noise)

# print(algorithm.create_data_test(df_italian, pn_list, im_list, fm_list, variant=variants.a_star))

# print(algorithm.create_data(log, pn, im, fm, variant=variants.a_star))
# print(algorithm.create_data(df_italian, inductive_pn_it_noise, inductive_im_it_noise, inductive_fm_it_noise, variant=variants.a_star))
# print(algorithm.create_data(df_italian, alpha_pn_it, alpha_im_it, alpha_fm_it, variant=variants.a_star))

# print(algorithm.apply_all_heuristics(log, pn, im, fm, variant=variants.a_star))
# print(algorithm.apply_log(df_bpi12, pn_bpi, im_bpi, fm_bpi, variant=variants.a_star))
# print(algorithm.apply_log(df_italian, pn_it, im_it, fm_it, variant=variants.a_star))
# print(algorithm.apply(df_problems, pn, im, fm, variant=variants.a_star))
