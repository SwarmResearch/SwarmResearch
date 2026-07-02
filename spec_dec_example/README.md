Example environment in `env/` for SwarmResearch to explore speculative decoding implementations. 

**Task**: Speculative decoding techniques are commonly lossless, where the target model verifies every draft token to ensure the output exactly matches what the target model would have generated. However, trading some accuracy for higher speed may be desirable. To explore this direction with SwarmResearch, our fast evaluator includes 30 total reasoning-intensive tasks from LiveCodeBench v6, AIME 2026, GPQA Diamond, and HLE-MCQ. The objective of SwarmResearch is to maximize token throughput while preserving benchmark accuracy. The baseline is a basic single-file, lossless speculative decoding implementation using rejection sampling.

Visualization of SwarmResearch's search:
<img src="../assets/evo_graph.png" alt="Evolution graph" width="900">