import flwr as fl
from typing import List, Tuple
from flwr.common import Metrics

global_round = 2

try:
    global_round = sys.argv[1]
    global_round = int(global_round)
except:
    pass

def weightef_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Initialize lists to store aggregated metrics for each label
    precision_list = []
    recall_list = []
    f1_score_list = []

    for label_idx in range(25):  # Assuming 25 classes
        label_precision = [num_examples * m["client_metrics"]["precision"][label_idx] for num_examples, m in metrics]
        label_recall = [num_examples * m["client_metrics"]["recall"][label_idx] for num_examples, m in metrics]
        label_f1_score = [num_examples * m["client_metrics"]["f1_score"][label_idx] for num_examples, m in metrics]

        precision = sum(label_precision) / sum([num_examples for num_examples, _ in metrics])
        recall = sum(label_recall) / sum([num_examples for num_examples, _ in metrics])
        f1_score = sum(label_f1_score) / sum([num_examples for num_examples, _ in metrics])

        # Append the aggregated metrics for each label
        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)

        # Print the aggregated metrics for each label
        print(f"Label {label_idx} - Precision: {precision}, Recall: {recall}, F1-Score: {f1_score}")

    # Aggregate the metrics for all labels
    aggregated_metrics = {
        "precision": sum(precision_list) / len(precision_list),
        "recall": sum(recall_list) / len(recall_list),
        "f1_score": sum(f1_score_list) / len(f1_score_list)
    }

    # Print the aggregated metrics for all labels
    print(f"Aggregated Metrics - Precision: {aggregated_metrics['precision']}, Recall: {aggregated_metrics['recall']}, F1-Score: {aggregated_metrics['f1_score']}")

    return aggregated_metrics

strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weightef_average,
    min_fit_clients=2,
    min_available_clients=2
)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=global_round),
    strategy=strategy
)

print("\nRound:", global_round)
