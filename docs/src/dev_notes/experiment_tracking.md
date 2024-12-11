# Technical Report on Experiment Tracking and Tool Analysis

## **Experiment Tracking Design**

To ensure robust and reproducible machine learning experiments, the tracking system must handle various experiment details and metrics effectively. Below is a detailed breakdown:

### **Experiment Details**

Each experiment should capture:

1. **Machine**: System specifications where the experiment is run.
	1. primeri (bgd-lab-16, nebulax1)
	- This is especily important when we have custom hardwere on each machine
3. **User**: User conducting the experiment.
4. **Experiment Name**: Identifier for the experiment that this run belongs to
5. **Hyperparameters/Architecture**: Detailed configurations of the model.
6. **Framework**: ML framework (e.g., pytorch, jax, forge-fe, tt-xla, ...).
7. **Code Version**: Git commit hash for reproducibility.
8. **Dataset Version**: Version of the dataset DVC?

### **Metrics to Track**

1. **Train/Validation Loss**
2. **Train/Validation Accuracy**
3. **Confusion Matrix**
4. **Time**: This is important when we want to improve performance
	1. todo: host/device time, subgraph execution time
5. **Memory Usage**
6. **Console Output**

### **Large Data Tracking**

Enable tracking of larger experiment artifacts:

1. **Histograms**: Useful for visualizing distributions (e.g., gradients, weights).
2. **Checkpoints**: Essential for resuming and analyzing models.
3. **Gradients**
4. **Optimizer State**
5. **Intermediates**
### **Architecture Requirements**

- **Database Integration**: A tracking server that connects to a database capable of handling large files via links to external storage.
- **Open Source**: Preference for tools that are freely accessible and modifiable.
- **Self-Hostable**: Deployment should not depend on external service providers.

## **Analysis of Potential Solutions**

### **Motivation for Experiment Tracking Tool Selection**

#TODO rewrite this a bit
In modern machine learning workflows, experiment tracking is critical for ensuring robust and reproducible experiments. The need for an effective tracking tool arises from the increasing complexity of ML projects, involving multiple team members, diverse datasets, and custom hardware. A comprehensive tracking solution must streamline experiment management, facilitate visualization, and integrate seamlessly with existing workflows.

Given the vast number of available tools, we focused on evaluating a subset that aligns with the following criteria:

2. **Experiment Management**: Support for logging metrics, parameters, and large artifacts.
3. **Visualization Capabilities**: Rich dashboards to analyze training metrics, histograms, and model states.
4. **Ease of Use**: Simple setup and integration with minimal dependencies.
5. **Specific Use Cases**: Tools tailored for tracking experiments rather than solely focusing on orchestration or production deployment.

To narrow down the options, we referred to some public list of tools for mlops as [Neptune.ai](https://neptune.ai/blog/best-open-source-mlops-tools) , [Awesome MLOps](https://github.com/awesome-mlops/awesome-ml-experiment-management?tab=readme-ov-file). Tools were grouped based on their primary use cases:
### **Grouping of Tools by Use Case**

#### **Experiment Tracking**
- **Weights & Biases (W&B)**
- **MLflow**
- **Aim**
- **TensorBoard**
- **ClearML**
- **Comet**

#### **Data Versioning**
- **DVC**
- **Pachyderm**
- **Keepsake**

#### **Orchestration**
- **Apache Airflow**
- **ZenML**
- **Argo Workflow**
- **MLRun**
- **Kedro**
- **Sematic**
#### **Production and Deployment**
- **Seldon Core**
- **EvidentlyAI**

## Closer

Here is a preview of features for some of the tools that were analysed:

| **Feature/Tool**            | **MLflow** | **Aim** | **Weights & Biases** | Tensorboard |
| --------------------------- | ---------- | ------- | -------------------- | ----------- |
| **Open Source**             | ✅          | ✅       | Not fully            | ✅           |
| **Self-Hostable**           | ✅          | ✅       | With price           | ✅           |
| **Scalar loggind**          | ✅          | ✅       | ✅                    | ✅           |
| **Histograms**              | ❌          | ✅       | ✅                    | ✅           |
| **Checkpoints**             | ✅          | ✅       | ✅                    | ❌           |
| **Console Output**          | ✅          | ✅       | ✅                    | ❌           |
| **Data Versioning**         | ✅          | ✅       | ✅                    | ❌           |
| **File reference support ** | ✅          | ✅       | ✅                    | ❌           |
| **Ease of self hosting**    | ✅          | ✅       | ❓                    | ✅           |

### **MLflow**

- **Strengths**: Comprehensive experiment tracking with robust integrations for logging metrics, parameters, and artefacts. Includes model versioning and serving capabilities. Open-source and self-hostable.
- **Weaknesses**: Visualisation capabilities are limited compared to Aim and W&B. Lacks built-in support for advanced histograms and distribution plots without external tools.

### **Aim**

- **Strengths**: Strong visualisation features for metrics and histograms. Open-source and self-hostable with minimal setup.
- **Weaknesses**: Comparatively less known that MLflow or Wandb. Requires additional setup for model versioning and serving, but this is not currently critical feature

### **Weights & Biases (W&B)**

- **Strengths**: Highly interactive dashboards for visualizations. Provides collaborative tools and powerful integrations for metrics tracking, checkpointing, and artifact refrencing.
- **Weaknesses**: Not fully open-source. Self-hosting is possible but comes with additional costs.

### **TensorBoard**

- **Strengths**: Industy standard for visulaziation of ml training pipeline.
- **Weaknesses**: Lacks advanced experiment management and model versioning capabilities. Checkpointing is not directly supported

## Conclusion

After analyzing the requirements and features of various experiment tracking tools, we decided to use **Weights & Biases (W&B)** for our projects. This decision was influenced by the following factors:

1. **Existing Integration**: W&B was already in use for some projects within TT, making it easier to adopt and integrate into our workflows without additional onboarding.
2. **Visualization Capabilities**: W&B offers excellent dashboards for interactive visualization, including support for histograms and advanced metrics, which are key for our needs.
3. **Collaborative Features**: The ability to share and collaborate on experiments seamlessly within a team was an added advantage.

While **Aim** presented itself as a strong contender due to its robust visualization capabilities and ease of self-hosting,
we could not allocate sufficient time to test and integrate it into our systems.
It remains a promising tool for future consideration, especially for its open-source flexibility and good visualisation.

**MLflow**, despite its comprehensive tracking features and integration with versioning systems,
was ruled out primarily because it does not natively support histogram visualisation, a critical requirement for our use case.
This missing feature limited its applicability for our experiment tracking and visualisation needs.

In summary, while W&B emerged as the most practical choice due to its existing usage and robust feature set,
Aim's potential as a self-hosted and open-source alternative should be explored further in the future.
