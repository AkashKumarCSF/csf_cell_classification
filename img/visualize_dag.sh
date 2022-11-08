#!/bin/bash
snakemake --dag all --configfile $1 > snakemake_workflow.graph

# Convert to JPG:
# cat snakemake_workflow.graph | dot -Tjpg > snakemake_workflow.jpg