function make_slurm_jobs(;
                         script_to_run="2-gtm-train.jl",
                         basename="gtm-k_",
                         n_tasks=4,
                         datapath="/scratch/jwaczak/data/robot-team/unsupervised",
                         # s = 0.5,
                         a = 0.1,
                         )

    job_name = basename

    file_text = """
    #!/bin/bash

    #SBATCH     --job-name=$(job_name)
    #SBATCH     --array=1-8
    #SBATCH     --output=$(job_name)_%a-%A.out
    #SBATCH     --error=$(job_name)_%a-%A.err
    #SBATCH     --nodes=1
    #SBATCH     --ntasks=1
    #SBATCH     --cpus-per-task=$(n_tasks)   # number of threads for multi-threading
    #SBATCH     --time=2-00:00:00
    #SBATCH     --mem=30G
    #SBATCH     --mail-type=ALL
    #SBATCH     --mail-user=jxw190004@utdallas.edu
    #SBATCH     --partition=normal

    julia --threads \$SLURM_CPUS_PER_TASK $(script_to_run) --datapath $(datapath) -s \$SLURM_ARRAY_TASK_ID -a $(a)
    """

    open("gtm-train__a-$(a).slurm", "w") do f
        println(f, file_text)
    end
end





for a in [0.001, 0.01, 0.1, 1.0, 10.0]
    make_slurm_jobs(
        a=a
    )
end


