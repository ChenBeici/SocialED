import logging
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score

class Metrics:
    @staticmethod
    def adjusted_rand_index(ground_truths, predicted_labels):
        """
        Calculate Adjusted Rand Index (ARI).
        """
        logging.info("Calculating Adjusted Rand Index (ARI)...")
        ari = adjusted_rand_score(ground_truths, predicted_labels)
        logging.info(f"Adjusted Rand Index (ARI): {ari}")
        return ari

    @staticmethod
    def adjusted_mutual_info(ground_truths, predicted_labels):
        """
        Calculate Adjusted Mutual Information (AMI).
        """
        logging.info("Calculating Adjusted Mutual Information (AMI)...")
        ami = adjusted_mutual_info_score(ground_truths, predicted_labels)
        logging.info(f"Adjusted Mutual Information (AMI): {ami}")
        return ami

    @staticmethod
    def normalized_mutual_info(ground_truths, predicted_labels):
        """
        Calculate Normalized Mutual Information (NMI).
        """
        logging.info("Calculating Normalized Mutual Information (NMI)...")
        nmi = normalized_mutual_info_score(ground_truths, predicted_labels)
        logging.info(f"Normalized Mutual Information (NMI): {nmi}")
        return nmi
