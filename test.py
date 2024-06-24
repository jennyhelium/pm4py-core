import pandas as pd
from multiprocessing import Process, Pool
import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking

from pm4py.algo.conformance.alignments.petri_net import algorithm
from pm4py.algo.conformance.alignments.petri_net import variants

import resource
import sys


def align(df, name_df, miner="alpha", noise_threshold=0.2, use_filter=True, top_k=1, use_k=True, cov=0.05, use_min_cov=True):
    print('Test ' + name_df + " " + miner + " " + str(noise_threshold))
    # discover petri net
    variance = 3
    print(name_df)

    if name_df == "prFm6":
        pn, im, fm = pm4py.read_pnml("pm4py/data/prFm6.pnml")
        fm = Marking()
        for p in pn.places:
            if p.name == 'n281':
                n281 = p

        fm[n281] = 1
        miner = "no"

    elif name_df == "prGm6":
        pn, im, fm = pm4py.read_pnml("pm4py/data/prGm6.pnml")
        fm = Marking()
        for p in pn.places:
            if p.name == 'n326':
                n326 = p
        fm[n326] = 1
        miner = "no"

    elif miner == "alpha":
        # alpha miner
        pn, im, fm = pm4py.discover_petri_net_alpha(df)

    elif miner == "inductive":
        # inductive miner
        if use_filter:
            print("Filter log")

            if not use_k:
                if use_min_cov:
                    # min_coverage
                    print("Min coverage", cov)
                    filtered_log = pm4py.filter_variants_by_coverage_percentage(df, cov)
                else:
                    # max_coverage
                    print("Max coverage", cov)
                    filtered_log = pm4py.filter_variants_by_maximum_coverage_percentage(df, cov)
            else:
                # top k
                k = top_k
                filtered_log = pm4py.filter_variants_top_k(df, k)

            pn, im, fm = pm4py.discover_petri_net_inductive(filtered_log)


            if not use_k:
                if use_min_cov:
                    pm4py.save_vis_petri_net(pn, im, fm,
                                             file_path="visualization/petri_net/" + name_df + "_filtered_" + miner + "_" +
                                                       str(variance) + ".png")
                else:
                    pm4py.save_vis_petri_net(pn, im, fm,
                                             file_path="visualization/petri_net/" + name_df + "_max_cov_filtered_" + miner + "_" +
                                                       str(variance) + ".png")
            else:
                pm4py.save_vis_petri_net(pn, im, fm, file_path="visualization/petri_net/" + name_df + "_top_" + str(k)
                                                               + "_" + miner + "_" + str(variance) + ".png")
        else:
            pn, im, fm = pm4py.discover_petri_net_inductive(df, noise_threshold=noise_threshold)


    elif miner == "heuristic":
        pn, im, fm = pm4py.discover_petri_net_heuristics(df, dependency_threshold=0.99)

    if len(fm) == 0:
        print("Empty final marking, create new final marking.")
        fm = Marking()

        for p in pn.places:
            if len(p.out_arcs) == 0:
                final_place = p

        fm[final_place] = 1

    print("Model discovered")

    variance = 3
    # miner = "no"

    # pm4py.save_vis_petri_net(pn, im, fm, file_path="visualization/petri_net/" + name_df + "_" + miner + "_" + str(
    #   variance) + ".png")

    print("Model saved")
    result = algorithm.create_data_pool(df, pn, im, fm, variance, name_df, miner, noise_threshold,
                                        variant=variants.a_star)
    if use_filter:
        if not use_k:
            if use_min_cov:
                result.to_pickle("results/" + name_df + "_filtered_" + miner + "_" + str(noise_threshold) + "_" + str(
                variance) + ".pkl")
            else:
                result.to_pickle("results/" + name_df + "_max_cov_filtered_" + miner + "_" + str(noise_threshold) + "_" + str(
                    variance) + ".pkl")
        else:
            result.to_pickle("results/" + name_df + "_top_" + str(k) + "_ext_" + miner + "_" + str(noise_threshold) + "_" +
                             str(variance) + ".pkl")
    else:
        result.to_pickle("results/" + name_df + "_ext_" + miner + "_" + str(noise_threshold) + "_" +
                         str(variance) + ".pkl")

    print(result)
    return result


def get_var_name(var):
    for name, value in globals().items():
        if value is var:
            return name

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
    print(resource.getrlimit(resource.RLIMIT_STACK))
    print(sys.getrecursionlimit())

    max_rec = 0x100000

    # May segfault without this line. 0x100 is a guess at the size of each stack frame.
    resource.setrlimit(resource.RLIMIT_STACK, [0x100 * max_rec, resource.RLIM_INFINITY])
    sys.setrecursionlimit(max_rec)

    #prFm6 = pm4py.read_xes("pm4py/data/prFm6.xes")
    #prGm6 = pm4py.read_xes("pm4py/data/prGm6.xes")

    #df_problems = pm4py.format_dataframe(pd.read_csv('pm4py/data/running_example_broken.csv', sep=';'),
     #                                   case_id='case:concept:name', activity_key='concept:name',
      #                                 timestamp_key='time:timestamp')

    bpi12 = pm4py.read_xes("pm4py/data/BPI_Challenge_2012.xes")
    road = pm4py.read_xes("pm4py/data/Road_Traffic_Fine_Management_Process.xes")
    sepsis = pm4py.read_xes("pm4py/data/Sepsis Cases - Event Log.xes")

    #permit = pm4py.read_xes("pm4py/data/PermitLog.xes")
    #prepaid = pm4py.read_xes("pm4py/data/PrepaidTravelCost.xes")
    #international_declaration = pm4py.read_xes("pm4py/data/InternationalDeclarations.xes")
    #request = pm4py.read_xes("pm4py/data/RequestForPayment.xes")
    #domestic = pm4py.read_xes("pm4py/data/DomesticDeclarations.xes")

    #data = [road, domestic, request, prepaid, international_declaration, sepsis, bpi12, permit]
    #data = [permit]
    data = [road, sepsis, bpi12]

    procs = []

    for d in data:
        name_df = get_var_name(d)
        #align(d, name_df, "inductive", noise_threshold=0, use_filter=True, use_min_cov=True)
        #align(d, name_df, "inductive", noise_threshold=0, use_filter=True, use_min_cov=False)
        align(d, name_df, "inductive", noise_threshold=0.2, use_filter=False, top_k=1, use_k=True, use_min_cov=False)
        #align(d, name_df, "inductive", noise_threshold=0, use_filter=True, top_k=5, use_k=True, use_min_cov=False)
        #align(d, name_df, "inductive", noise_threshold=0, use_filter=True, top_k=10, use_k=True, use_min_cov=False)

    print("Done")


# print(algorithm.create_data(df_sepsis, pn_im_sepsis, im_im_sepsis, fm_im_sepsis, variant=variants.a_star))

# log = pm4py.convert_to_event_log(df_problems)

# print(algorithm.create_data(log, pn, im, fm, variant=variants.a_star))
# print(algorithm.apply_all_heuristics(log, pn, im, fm, variant=variants.a_star))
