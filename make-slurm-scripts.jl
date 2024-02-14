function make_slurm_jobs(;
                         script_to_run="2-gtm-train.jl",
                         basename="gtm-k_",
                         n_tasks=4,
                         datapath="/scratch/jwaczak/data/robot-team/unsupervised",
                         m_max = 50,
                         s = 0.5,
                         a = 0.1,
                         refs_only = true,
                         )

    job_name = basename

    file_text = """
    #!/bin/bash

    #SBATCH     --job-name=$(job_name)
    #SBATCH     --array=2-50
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

    julia --threads \$SLURM_CPUS_PER_TASK $(script_to_run) --datapath $(datapath) -k \$SLURM_ARRAY_TASK_ID -m $(m_max) -s $(s) -a $(a)
    """

    open("gtm-train__s-$(s)__a-$(a).slurm", "w") do f
        println(f, file_text)
    end
end





for a in [0.01, 0.1, 0.5, 1.0]
    for s in [0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
	make_slurm_jobs(
			s=s,
			a=a 
			)
    end
end


