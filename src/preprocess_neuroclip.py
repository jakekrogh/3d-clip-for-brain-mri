import dotenv
dotenv.load_dotenv()

from planning.YuccaClipPlanner import YuccaClipPlanner


def main():
    task = "Task001_GammaKnife"
    preprocessor_name = "YuccaClipPreprocessor"
    disable_sanity_checks = True
    enable_cc_analysis = True
    threads = 8
    planner = YuccaClipPlanner(task, preprocessor_name, threads=threads, disable_sanity_checks=disable_sanity_checks, enable_cc_analysis=enable_cc_analysis)
    planner.plan()
    planner.preprocess()


if __name__ == "__main__":
    main()
 