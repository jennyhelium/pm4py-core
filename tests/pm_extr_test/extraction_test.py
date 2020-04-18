import inspect
import os
import sys
import traceback

if __name__ == "__main__":
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    parentdir2 = os.path.dirname(parentdir)
    sys.path.insert(0, parentdir)
    sys.path.insert(0, parentdir2)
    import time
    from pm4py.objects.log.importer.xes import factory as xes_factory
    from pm4py.algo.discovery.inductive import factory as inductive
    from pm4py.algo.conformance.alignments import factory as align_factory
    from pm4py.algo.discovery.alpha import factory as alpha
    from pm4py.algo.discovery.heuristics import factory as heuristics_miner
    from pm4py.objects.petri import check_soundness
    from pm4py.evaluation.replay_fitness import factory as fitness_factory
    from pm4py.evaluation.precision import factory as precision_factory
    from pm4py.evaluation.simplicity import factory as simplicity_factory
    from pm4py.evaluation.generalization import factory as generalization_factory
    from pm4py.objects.log.util import insert_classifier
    from pm4py.objects.petri.exporter import pnml as pnml_exporter
    from pm4py.visualization.petrinet import factory as petri_vis_factory
    from pm4py.visualization.common.save import save as vis_save
    from pm4py import util as pmutil


    def get_elonged_string(stru):
        nchar = 30

        if len(stru) >= nchar:
            return stru

        return stru + " ".join([""] * (nchar - len(stru)))


    def get_elonged_float(value):
        stru = "%.3f" % value

        return get_elonged_string(stru)

    ENABLE_VISUALIZATIONS = False
    ENABLE_VISUALIZATIONS_INDUCTIVE = False
    ENABLE_ALIGNMENTS = False
    ENABLE_PRECISION = False
    ENABLE_PETRI_EXPORTING = False
    CHECK_SOUNDNESS = False
    align_factory.DEFAULT_VARIANT = align_factory.VERSION_DIJKSTRA_NO_HEURISTICS
    logFolder = os.path.join("..", "compressed_input_data")
    pnmlFolder = "pnml_folder"
    pngFolder = "png_folder"
    times_tokenreplay_alpha = {}
    times_tokenreplay_imdf = {}
    times_alignments_imdf = {}
    fitness_token_alpha = {}
    fitness_token_imdf = {}
    fitness_align_imdf = {}
    precision_alpha = {}
    precision_imdf = {}
    simplicity_alpha = {}
    simplicity_imdf = {}
    generalization_alpha = {}
    generalization_imdf = {}


    def write_report():
        f = open("report.txt", "w")

        f.write("\n\n")
        f.write("Fitness on Alpha and Inductive models - measured by token-based replay and alignments\n")
        f.write("----\n")
        f.write(
            get_elonged_string("log") + "\t" + get_elonged_string("fitness_token_alpha") + "\t" + get_elonged_string(
                "times_tokenreplay_alpha") + "\t" + get_elonged_string(
                "fitness_token_imdf") + "\t" + get_elonged_string("times_tokenreplay_imdf"))
        if ENABLE_ALIGNMENTS:
            f.write(
                "\t" + get_elonged_string("fitness_align_imdf") + "\t" + get_elonged_string("times_alignments_imdf"))
        f.write("\n")
        for this_logname in precision_alpha:
            # F.write("%s\t\t%.3f\t\t%.3f\n" % (logName, fitness_token_alpha[logName], fitness_token_imdf[logName]))
            f.write(get_elonged_string(this_logname))
            f.write("\t")
            f.write(get_elonged_float(fitness_token_alpha[this_logname]))
            f.write("\t")
            f.write(get_elonged_float(times_tokenreplay_alpha[this_logname]))
            f.write("\t")
            f.write(get_elonged_float(fitness_token_imdf[this_logname]))
            f.write("\t")
            f.write(get_elonged_float(times_tokenreplay_imdf[this_logname]))
            if ENABLE_ALIGNMENTS:
                f.write("\t")
                f.write(get_elonged_float(fitness_align_imdf[this_logname]))
                f.write("\t")
                f.write(get_elonged_float(times_alignments_imdf[this_logname]))
            f.write("\n")
        f.write("\n\n")
        f.write("Precision measured by ETConformance where activated transitions are retrieved using token replay\n")
        f.write("----\n")
        f.write(get_elonged_string("log") + "\t" + get_elonged_string("precision_alpha") + "\t" + get_elonged_string(
            "precision_imdf") + "\n")
        for this_logname in precision_alpha:
            f.write(get_elonged_string(this_logname))
            f.write("\t")
            f.write(get_elonged_float(precision_alpha[this_logname]))
            f.write("\t")
            f.write(get_elonged_float(precision_imdf[this_logname]))
            f.write("\n")
        f.write("\n\n")
        f.write("Generalization based on token replay transition recall\n")
        f.write("----\n")
        f.write(
            get_elonged_string("log") + "\t" + get_elonged_string("generalization_alpha") + "\t" + get_elonged_string(
                "generalization_imdf") + "\n")
        for this_logname in precision_alpha:
            f.write(get_elonged_string(this_logname))
            f.write("\t")
            f.write(get_elonged_float(generalization_alpha[this_logname]))
            f.write("\t")
            f.write(get_elonged_float(generalization_imdf[this_logname]))
            f.write("\n")
        f.write("\n\n")
        f.write("Simplicity based on inverse arc degree\n")
        f.write("----\n")
        f.write(get_elonged_string("log") + "\t" + get_elonged_string("simplicity_alpha") + "\t" + get_elonged_string(
            "simplicity_imdf") + "\n")
        for this_logname in precision_alpha:
            f.write(get_elonged_string(this_logname))
            f.write("\t")
            f.write(get_elonged_float(simplicity_alpha[this_logname]))
            f.write("\t")
            f.write(get_elonged_float(simplicity_imdf[this_logname]))
            f.write("\n")
        f.write("\n")
        f.close()


    for logName in os.listdir(logFolder):
        if "." in logName:
            logNamePrefix = logName.split(".")[0]
            logExtension = logName[len(logNamePrefix)+1:]

            print("\nelaborating " + logName)

            logPath = os.path.join(logFolder, logName)
            if "xes" in logExtension:
                log = xes_factory.import_log(logPath, variant="iterparse")
            else:
                from pm4py.objects.log.importer.parquet import factory as parquet_importer
                from pm4py.objects.conversion.log import factory as log_conv_factory
                dataframe = parquet_importer.apply(logPath)
                log = log_conv_factory.apply(dataframe)
                del dataframe

            log, classifier_key = insert_classifier.search_act_class_attr(log, force_activity_transition_insertion=True)

            print("loaded log")

            activity_key = "concept:name"
            if classifier_key is not None:
                activity_key = classifier_key

            parameters_discovery = {pmutil.constants.PARAMETER_CONSTANT_ACTIVITY_KEY: activity_key,
                                    pmutil.constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: activity_key}
            t1 = time.time()
            alpha_model, alpha_initial_marking, alpha_final_marking = alpha.apply(log, parameters=parameters_discovery)
            if ENABLE_PETRI_EXPORTING:
                pnml_exporter.export_net(alpha_model, alpha_initial_marking,
                                         os.path.join(pnmlFolder, logNamePrefix + "_alpha.pnml"),
                                         final_marking=alpha_final_marking)
            t2 = time.time()
            print("time interlapsed for calculating Alpha Model", (t2 - t1))
            if CHECK_SOUNDNESS:
                print("alpha is_sound_wfnet", check_soundness.check_petri_wfnet_and_soundness(alpha_model, debug=True))

            t1 = time.time()
            heu_model, heu_initial_marking, heu_final_marking = heuristics_miner.apply(log,
                                                                                       parameters=parameters_discovery)
            if ENABLE_PETRI_EXPORTING:
                pnml_exporter.export_net(heu_model, heu_initial_marking,
                                         os.path.join(pnmlFolder, logNamePrefix + "_alpha.pnml"),
                                         final_marking=heu_final_marking)
            t2 = time.time()
            print("time interlapsed for calculating Heuristics Model", (t2 - t1))
            if CHECK_SOUNDNESS:
                print("heuristics is_sound_wfnet", check_soundness.check_petri_wfnet_and_soundness(heu_model, debug=True))

            t1 = time.time()
            tree = inductive.apply_tree(log, parameters=parameters_discovery)
            print(tree)
            inductive_model, inductive_im, inductive_fm = inductive.apply(log, parameters=parameters_discovery)
            if ENABLE_PETRI_EXPORTING:
                pnml_exporter.export_net(inductive_model, inductive_im,
                                         os.path.join(pnmlFolder, logNamePrefix + "_inductive.pnml"),
                                         final_marking=inductive_fm)
            """
            generated_log = pt_semantics.generate_log(tree)
            print("first trace of log", [x["concept:name"] for x in generated_log[0]])
            """
            t2 = time.time()
            print("time interlapsed for calculating Inductive Model", (t2 - t1))
            if CHECK_SOUNDNESS:
                print("inductive is_sound_wfnet",
                      check_soundness.check_petri_wfnet_and_soundness(inductive_model, debug=True))

            parameters = {pmutil.constants.PARAMETER_CONSTANT_ACTIVITY_KEY: activity_key,
                          pmutil.constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: activity_key, "format": "png"}

            t1 = time.time()
            fitness_token_alpha[logName] = \
                fitness_factory.apply(log, alpha_model, alpha_initial_marking, alpha_final_marking,
                                      parameters=parameters, variant="token_replay")[
                    'perc_fit_traces']
            print(str(time.time())+" fitness_token_alpha for " + logName + " succeeded! "+str(fitness_token_alpha[logName]))
            t2 = time.time()
            times_tokenreplay_alpha[logName] = t2 - t1

            t1 = time.time()
            fitness_token_imdf[logName] = \
                fitness_factory.apply(log, inductive_model, inductive_im, inductive_fm, parameters=parameters,
                                      variant="token_replay")[
                    'perc_fit_traces']
            print(str(time.time())+" fitness_token_inductive for " + logName + " succeeded! "+str(fitness_token_imdf[logName]))
            t2 = time.time()
            times_tokenreplay_imdf[logName] = t2 - t1

            if ENABLE_ALIGNMENTS:
                t1 = time.time()
                fitness_align_imdf[logName] = \
                    fitness_factory.apply(log, inductive_model, inductive_im, inductive_fm,
                                          variant="alignments", parameters=parameters)['percFitTraces']
                print(str(time.time())+" fitness_token_align for " + logName + " succeeded! "+str(fitness_align_imdf[logName]))
                t2 = time.time()
                times_alignments_imdf[logName] = t2 - t1

            if ENABLE_PRECISION:
                precision_alpha[logName] = precision_factory.apply(log, alpha_model, alpha_initial_marking,
                                                                   alpha_final_marking, variant="etconformance", parameters=parameters)
            else:
                precision_alpha[logName] = 0.0
            print(str(time.time())+" precision_alpha for " + logName + " succeeded! "+str(precision_alpha[logName]))

            generalization_alpha[logName] = generalization_factory.apply(log, alpha_model, alpha_initial_marking,
                                                                         alpha_final_marking, parameters=parameters)
            print(str(time.time())+" generalization_alpha for " + logName + " succeeded! "+str(generalization_alpha[logName]))
            simplicity_alpha[logName] = simplicity_factory.apply(alpha_model, parameters=parameters)
            print(str(time.time())+" simplicity_alpha for " + logName + " succeeded! "+str(simplicity_alpha[logName]))

            if ENABLE_PRECISION:
                precision_imdf[logName] = precision_factory.apply(log, inductive_model, inductive_im,
                                                              inductive_fm, variant="etconformance", parameters=parameters)
            else:
                precision_imdf[logName] = 0.0
            print(str(time.time())+" precision_imdf for " + logName + " succeeded! "+str(precision_imdf[logName]))

            generalization_imdf[logName] = generalization_factory.apply(log, inductive_model, inductive_im,
                                                                        inductive_fm, parameters=parameters)
            print(str(time.time())+" generalization_imdf for " + logName + " succeeded! "+str(generalization_imdf[logName]))
            simplicity_imdf[logName] = simplicity_factory.apply(inductive_model, parameters=parameters)
            print(str(time.time())+" simplicity_imdf for " + logName + " succeeded! "+str(simplicity_imdf[logName]))

            write_report()

            if ENABLE_VISUALIZATIONS:
                try:
                    alpha_vis = petri_vis_factory.apply(alpha_model, alpha_initial_marking, alpha_final_marking, log=log,
                                                        parameters=parameters, variant="frequency")
                    vis_save(alpha_vis, os.path.join(pngFolder, logNamePrefix + "_alpha.png"))
                    print(str(time.time())+" alpha visualization for "+logName+" succeeded!")
                except:
                    print(str(time.time())+" alpha visualization for "+logName+" failed!")
                    traceback.print_exc()

                try:
                    heuristics_vis = petri_vis_factory.apply(heu_model, heu_initial_marking, heu_final_marking,
                                                             log=log, parameters=parameters, variant="frequency")
                    vis_save(heuristics_vis, os.path.join(pngFolder, logNamePrefix + "_heuristics.png"))
                    print(str(time.time())+" heuristics visualization for "+logName+" succeeded!")
                except:
                    print(str(time.time())+" heuristics visualization for " + logName + " failed!")
                    traceback.print_exc()

            if ENABLE_VISUALIZATIONS or ENABLE_VISUALIZATIONS_INDUCTIVE:
                try:
                    inductive_vis = petri_vis_factory.apply(inductive_model, inductive_im, inductive_fm,
                                                            log=log, parameters=parameters, variant="frequency")
                    vis_save(inductive_vis, os.path.join(pngFolder, logNamePrefix + "_inductive.png"))
                    print(str(time.time())+" inductive visualization for "+logName+" succeeded!")
                except:
                    print(str(time.time())+" inductive visualization for " + logName + " failed!")
                    traceback.print_exc()