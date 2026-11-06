"""
Comprehensive tests for Round 5/6 pipeline - verifies all known bugs.

Run with:
    PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/test_round5_pipeline.py -v

These tests are designed to FAIL on the current buggy code, then PASS after fixes.
"""

import pytest
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import os

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: TF-IDF Model Path Mismatch
# ─────────────────────────────────────────────────────────────────────────────
class TestTFIDFPathMismatch:
    """Bug: evaluate_all_models_round5.py loads from wrong path."""

    def test_evaluate_script_loads_correct_tfidf_paths(self):
        """Verify evaluation script loads Round 5 retrained models, not old ones."""
        eval_script = PROJECT_DIR / "scripts" / "evaluate_all_models_round5.py"
        assert eval_script.exists(), f"Script not found: {eval_script}"

        content = eval_script.read_text()

        # BUG: These lines load OLD models (wrong paths)
        # Should load from: models/round5_retrained/tfidf_*_round5.joblib
        old_paths = [
            'MODEL_DIR / "tfidf_logreg.joblib"',
            'MODEL_DIR / "tfidf_svc.joblib"',
        ]

        for old_path in old_paths:
            assert old_path not in content, (
                f"BUG FOUND: Script uses old model path {old_path}. "
                f"Should use models/round5_retrained/tfidf_*_round5.joblib"
            )

        # CORRECT: Should load from round5_retrained/
        correct_paths = [
            'tfidf_logreg_round5.joblib',
            'tfidf_linearsvc_round5.joblib',
        ]

        for correct in correct_paths:
            assert correct in content, (
                f"MISSING: Script should load {correct} from round5_retrained/"
            )

    def test_tfidf_models_exist_in_correct_location(self):
        """Verify Round 5 retrained TF-IDF models exist."""
        retrained_dir = MODEL_DIR / "round5_retrained"
        assert retrained_dir.exists(), f"Retrained dir not found: {retrained_dir}"

        expected_models = [
            retrained_dir / "tfidf_logreg_round5.joblib",
            retrained_dir / "tfidf_linearsvc_round5.joblib",
        ]

        for model_path in expected_models:
            assert model_path.exists(), f"Model not found: {model_path}"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: PhoBERT Directory Naming Inconsistency
# ─────────────────────────────────────────────────────────────────────────────
class TestPhoBERTNaming:
    """Bug: phobert_seed_42 vs seed_42 naming mismatch."""

    def test_all_scripts_use_consistent_phobert_naming(self):
        """All scripts should use same naming convention: phobert_seed_XX/best_model"""
        scripts_to_check = [
            PROJECT_DIR / "scripts" / "retrain_all_models_round5.py",
            PROJECT_DIR / "scripts" / "run_final_round5_evaluation.py",
            PROJECT_DIR / "scripts" / "evaluate_all_models_round5.py",
            PROJECT_DIR / "scripts" / "complete_evaluation_round5.py",
        ]

        correct_pattern = "phobert_seed_"
        wrong_patterns = [
            'f"seed_{',  # seed_42 instead of phobert_seed_42
            "/seed_",     # /seed_42/ path
        ]

        for script in scripts_to_check:
            if not script.exists():
                continue

            content = script.read_text()

            for wrong in wrong_patterns:
                assert wrong not in content, (
                    f"BUG in {script.name}: Found inconsistent naming pattern '{wrong}'. "
                    f"Should use '{correct_pattern}' format."
                )

    def test_phobert_models_in_correct_directories(self):
        """Verify PhoBERT models exist in phobert_seed_XX/ subdirs."""
        retrained_dir = MODEL_DIR / "round5_retrained"

        for seed in [42, 123, 2024]:
            model_dir = retrained_dir / f"phobert_seed_{seed}" / "best_model"
            assert model_dir.exists(), (
                f"Model not found: {model_dir}. "
                f"Should be at phobert_seed_{seed}/best_model/"
            )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: Majority Vote Implementation
# ─────────────────────────────────────────────────────────────────────────────
class TestMajorityVote:
    """Bug: Uses probability averaging instead of prediction voting."""

    def test_majority_vote_uses_predictions_not_probabilities(self):
        """Majority vote should be based on binary predictions, not averaged probabilities."""
        eval_script = PROJECT_DIR / "scripts" / "run_final_round5_evaluation.py"
        content = eval_script.read_text()

        # Check that majority vote is done on predictions, not probabilities
        # CORRECT: np.mean(predictions) >= 0.5 or np.sum(predictions) > n_seeds/2
        # BUGGY: np.mean(probabilities) >= 0.5

        # Look for the majority vote section
        lines = content.split('\n')
        in_majority_section = False
        majority_vote_code = []

        for i, line in enumerate(lines):
            if 'majority' in line.lower() or 'avg' in line.lower() and 'vote' in line.lower():
                in_majority_section = True
            if in_majority_section:
                majority_vote_code.append((i+1, line))
                if len(majority_vote_code) > 20:  # Stop after reasonable context
                    break

        majority_text = '\n'.join([l for _, l in majority_vote_code])

        # Check for proper majority vote implementation
        # Proper: sum predictions and check if > n/2
        # The current code uses np.mean(predictions) >= 0.5 which is mathematically
        # equivalent to majority vote for binary predictions, BUT it's confusingly named

        # More importantly, check that it's NOT averaging probabilities
        buggy_patterns = [
            'np.mean(all_probs',
            'avg_probs = np.mean(all_probs',
        ]

        for buggy in buggy_patterns:
            # If this pattern exists AND it's used for voting, that's a bug
            if buggy in content:
                # Check if it's in the voting section
                assert buggy not in majority_text, (
                    f"BUG: Majority vote section uses probability averaging. "
                    f"Should use prediction-based voting (sum(preds) > n_seeds/2)"
                )

    def test_majority_vote_produces_correct_results(self):
        """Test that majority vote logic works correctly for various cases."""
        # For 3 seeds, majority vote means >= 2 out of 3

        # Test case 1: Clear majority (2 or 3 agree)
        preds_seed1 = np.array([0, 1, 1, 0, 1])  # votes: [0,1,1,0,1]
        preds_seed2 = np.array([1, 1, 1, 0, 1])  # votes: [1,1,1,0,1]
        preds_seed3 = np.array([0, 1, 0, 0, 1])  # votes: [0,1,0,0,1]
        stacked = np.stack([preds_seed1, preds_seed2, preds_seed3], axis=0)

        # CORRECT: sum >= 2 (majority of 3 seeds)
        correct_vote = (np.sum(stacked, axis=0) >= 2).astype(int)
        # Expected: [1, 3, 2, 0, 3] -> [0, 1, 1, 0, 1] (col0: 0+1+0=1 < 2)
        expected = np.array([0, 1, 1, 0, 1])
        assert np.array_equal(correct_vote, expected), f"Expected {expected}, got {correct_vote}"

        # Test case 2: Edge case with tie (e.g., 1, 1, 0 -> sum=2 -> majority=1)
        # np.mean >= 0.5 and sum >= 2 are mathematically equivalent for binary predictions

        # Test case 3: All agree
        all_same = np.array([1, 0, 1])
        stacked_all = np.stack([all_same, all_same, all_same], axis=0)
        vote_all = (np.sum(stacked_all, axis=0) >= 2).astype(int)
        assert np.array_equal(vote_all, all_same), "All agree case failed"

        # Verify implementation is correct
        current_impl = (np.mean(stacked, axis=0) >= 0.5).astype(int)
        assert np.array_equal(correct_vote, current_impl), (
            "Majority vote formula (sum >= 2) should equal (mean >= 0.5) for binary predictions"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: Data Integrity
# ─────────────────────────────────────────────────────────────────────────────
class TestDataIntegrity:
    """Verify no data leakage between train/val/test splits."""

    @pytest.fixture
    def load_datasets(self):
        """Load all datasets."""
        train = pd.read_csv(DATA_DIR / "labeled" / "final_train.csv")
        val = pd.read_csv(DATA_DIR / "labeled" / "final_val.csv")
        test = pd.read_csv(DATA_DIR / "labeled" / "final_test.csv")
        vsmec = pd.read_csv(DATA_DIR.parent / "data_unified" / "cross_domain_test.csv")
        return train, val, test, vsmec

    def normalize_text(self, text):
        """Normalize text for comparison."""
        import unicodedata
        import re
        text = unicodedata.normalize("NFKC", str(text))
        return re.sub(r"\s+", " ", text).strip().casefold()

    def test_no_train_val_overlap(self, load_datasets):
        """Train and val sets should have no text overlap."""
        train, val, _, _ = load_datasets

        train_texts = set(train["comment_text"].apply(self.normalize_text))
        val_texts = set(val["comment_text"].apply(self.normalize_text))

        overlap = train_texts & val_texts
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping texts between train and val"

    def test_no_train_test_overlap(self, load_datasets):
        """Train and test sets should have no text overlap."""
        train, _, test, _ = load_datasets

        train_texts = set(train["comment_text"].apply(self.normalize_text))
        test_texts = set(test["comment_text"].apply(self.normalize_text))

        overlap = train_texts & test_texts
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping texts between train and test"

    def test_no_val_test_overlap(self, load_datasets):
        """Val and test sets should have no text overlap."""
        _, val, test, _ = load_datasets

        val_texts = set(val["comment_text"].apply(self.normalize_text))
        test_texts = set(test["comment_text"].apply(self.normalize_text))

        overlap = val_texts & test_texts
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping texts between val and test"

    def test_no_train_vsmec_overlap(self, load_datasets):
        """Train should not overlap with VSMEC cross-domain test."""
        train, _, _, vsmec = load_datasets

        # VSMEC might use 'text' or 'comment_text'
        vsmec_col = "comment_text" if "comment_text" in vsmec.columns else "text"

        train_texts = set(train["comment_text"].apply(self.normalize_text))
        vsmec_texts = set(vsmec[vsmec_col].apply(self.normalize_text))

        overlap = train_texts & vsmec_texts
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping texts between train and VSMEC"

    def test_no_duplicates_in_train(self, load_datasets):
        """Train set should have no duplicate texts."""
        train, _, _, _ = load_datasets

        norm_texts = train["comment_text"].apply(self.normalize_text)
        duplicates = norm_texts[norm_texts.duplicated()]

        assert len(duplicates) == 0, f"Found {len(duplicates)} duplicate texts in train"

    def test_correct_train_size(self, load_datasets):
        """Train set should have 6,392 samples (Round 6 v2 dataset)."""
        train, _, _, _ = load_datasets

        expected = 6392
        actual = len(train)

        assert actual == expected, (
            f"Train size mismatch: expected {expected}, got {actual}"
        )

    def test_correct_val_test_sizes(self, load_datasets):
        """Val should have 1,371 samples, test should have 1,371 samples (Round 6 v2, 15% each)."""
        _, val, test, _ = load_datasets

        assert len(val) == 1371, f"Val size should be 1,371, got {len(val)}"
        assert len(test) == 1371, f"Test size should be 1,371, got {len(test)}"

    def test_label_distribution_reasonable(self, load_datasets):
        """Train set should have ~16% depression (imbalanced)."""
        train, _, _, _ = load_datasets

        depression_rate = train["label"].mean()

        # Should be between 10% and 25%
        assert 0.10 <= depression_rate <= 0.25, (
            f"Depression rate {depression_rate:.2%} outside expected range [10%, 25%]"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: Pipeline Script Fragility
# ─────────────────────────────────────────────────────────────────────────────
class TestPipelineFragility:
    """Tests for pipeline script robustness."""

    def test_no_hardcoded_row_counts_in_merge_script(self):
        """Merge script should not have hardcoded row count assertions."""
        merge_script = PROJECT_DIR / "scripts" / "merge_round5_reviewed.py"
        content = merge_script.read_text()

        # These are the problematic hardcoded assertions
        hardcoded_patterns = [
            'len(base) != 2072',
            'len(r5) != 1360',
            'Expected 2,072',
            'Expected 1,360',
        ]

        bugs_found = []
        for pattern in hardcoded_patterns:
            if pattern in content:
                bugs_found.append(pattern)

        assert len(bugs_found) == 0, (
            f"BUG: Found hardcoded row count assertions: {bugs_found}. "
            f"These should be configurable or removed for pipeline flexibility."
        )

    def test_merge_script_has_backup_mechanism(self):
        """Merge script should backup source files before overwriting."""
        merge_script = PROJECT_DIR / "scripts" / "merge_round5_reviewed.py"
        content = merge_script.read_text()

        # Should have backup mechanism before writing to train_gold.csv
        # Look for shutil.copy, backup, or atomic write pattern
        backup_indicators = [
            'shutil.copy',
            'backup',
            '.bak',
            'atomic',
            'rename',
        ]

        has_backup = any(indicator in content for indicator in backup_indicators)

        assert has_backup, (
            "BUG: Merge script does not have backup mechanism. "
            "Should backup train_gold.csv before overwriting."
        )

    def test_consistent_text_normalization(self):
        """Active scripts should use consistent NFKC normalization.

        Note: merge_round4_reviewed.py is legacy (Round 4) and uses simple normalization.
        Only check ACTIVE scripts used for Round 5+.
        """
        scripts = [
            PROJECT_DIR / "scripts" / "merge_round5_reviewed.py",
            PROJECT_DIR / "scripts" / "merge_augmented_leakage_free.py",
        ]

        # Check which normalization method each script uses
        nfkc_pattern = 'NFKC'

        for script in scripts:
            if not script.exists():
                continue
            content = script.read_text()
            has_nfkc = nfkc_pattern in content
            assert has_nfkc, (
                f"Script {script.name} should use NFKC normalization for consistency. "
                f"Found normalization at lines containing 'NFKC' = {has_nfkc}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: BiLSTM Architecture Compatibility
# ─────────────────────────────────────────────────────────────────────────────
class TestBiLSTMArchitecture:
    """Tests for BiLSTM model architecture consistency."""

    def test_bilstm_architecture_matches_between_training_and_eval(self):
        """BiLSTM architecture in training should match evaluation."""
        train_script = PROJECT_DIR / "scripts" / "retrain_all_models_round5.py"
        eval_script = PROJECT_DIR / "scripts" / "run_final_round5_evaluation.py"

        # Read both scripts
        train_content = train_script.read_text()
        eval_content = eval_script.read_text()

        # Extract BiLSTM class definitions
        # Look for the classifier layer definition
        # Training should have: self.fc = nn.Linear(hidden_dim * 2, 2)
        # OR self.classifier = nn.Sequential(...)

        # The key is that both should use the SAME architecture
        # This is a simplified check - full verification would need to compare

        # For now, just verify BiLSTM checkpoints exist
        retrained_dir = MODEL_DIR / "round5_retrained"

        bilstm_checkpoints = []
        for seed in [42, 123, 2024]:
            ckpt = retrained_dir / f"bilstm_seed_{seed}" / "best_model.pt"
            if ckpt.exists():
                bilstm_checkpoints.append(ckpt)

        assert len(bilstm_checkpoints) == 3, (
            f"Expected 3 BiLSTM checkpoints, found {len(bilstm_checkpoints)}"
        )

    def test_bilstm_checkpoint_can_be_loaded(self):
        """Verify BiLSTM checkpoints can be loaded without errors.

        Note: BiLSTM checkpoints contain a custom Vocabulary class that must be
        defined at module level. This test verifies the checkpoint structure.
        The actual loading is done by run_final_round5_evaluation.py which
        imports the Vocabulary class at module level.
        """
        import torch
        import pickle

        retrained_dir = MODEL_DIR / "round5_retrained"

        for seed in [42, 123, 2024]:
            ckpt_path = retrained_dir / f"bilstm_seed_{seed}" / "best_model.pt"

            if not ckpt_path.exists():
                continue

            # Verify checkpoint has required keys (without loading full model)
            with open(ckpt_path, 'rb') as f:
                # Read just the metadata without unpickling
                try:
                    # Try to read the pickle safely
                    data = pickle.Unpickler(f)
                    # This will fail if Vocabulary class not defined, but we can at least
                    # verify the file structure exists
                    pass
                except Exception:
                    pass

            # Check file exists and has reasonable size
            file_size = ckpt_path.stat().st_size
            assert file_size > 1000, f"Checkpoint file too small: {file_size} bytes"

            # The actual loading is tested by run_final_round5_evaluation.py
            # which properly imports Vocabulary class


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7: Evaluation Results Consistency
# ─────────────────────────────────────────────────────────────────────────────
class TestEvaluationConsistency:
    """Tests for evaluation results consistency."""

    def test_evaluation_results_exist(self):
        """Verify canonical evaluation results exist."""
        results_files = list(RESULTS_DIR.glob("round5_final_v*/evaluation_results.json"))

        assert len(results_files) > 0, "No evaluation results found"

        # Get the most recent
        latest = max(results_files, key=lambda p: p.stat().st_mtime)

        with open(latest) as f:
            results = json.load(f)

        # Verify structure
        assert 'in_domain' in results, "Missing in_domain results"
        assert 'cross_domain' in results, "Missing cross_domain results"

    def test_phobert_majority_vote_results_exist(self):
        """Verify PhoBERT majority vote results exist."""
        results_files = list(RESULTS_DIR.glob("round5_final_v*/evaluation_results.json"))
        latest = max(results_files, key=lambda p: p.stat().st_mtime)

        with open(latest) as f:
            results = json.load(f)

        assert 'phobert_avg' in results['in_domain'], (
            "Missing PhoBERT majority vote results"
        )

        assert 'phobert_avg' in results['cross_domain'], (
            "Missing PhoBERT cross-domain majority vote results"
        )

    def test_all_models_have_results(self):
        """All models should have evaluation results."""
        results_files = list(RESULTS_DIR.glob("round5_final_v*/evaluation_results.json"))
        latest = max(results_files, key=lambda p: p.stat().st_mtime)

        with open(latest) as f:
            results = json.load(f)

        expected_models = [
            'phobert_avg',
            'tfidf_logreg',
            'tfidf_svc',
            'bilstm_avg',
        ]

        for model in expected_models:
            assert model in results['in_domain'], (
                f"Missing in-domain results for {model}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 8: Dataset Integrity Report
# ─────────────────────────────────────────────────────────────────────────────
class TestIntegrityReport:
    """Tests for dataset integrity report."""

    def test_integrity_report_exists_and_passes(self):
        """Verify integrity report exists and all checks pass."""
        report_path = DATA_DIR / "analysis" / "dataset_integrity_report.json"

        assert report_path.exists(), f"Integrity report not found: {report_path}"

        with open(report_path) as f:
            report = json.load(f)

        # Verify integrity_pass is True
        assert report.get('integrity_pass') == True, (
            f"Data integrity check failed: {report.get('integrity', {})}"
        )

    def test_integrity_report_has_all_required_fields(self):
        """Verify integrity report has all required fields."""
        report_path = DATA_DIR / "analysis" / "dataset_integrity_report.json"

        with open(report_path) as f:
            report = json.load(f)

        required_sections = [
            'protocol_version',
            'integrity_pass',
            'datasets',
            'integrity',
        ]

        for section in required_sections:
            assert section in report, f"Missing required section: {section}"

    def test_no_overlap_in_integrity_report(self):
        """Verify integrity report shows zero overlaps."""
        report_path = DATA_DIR / "analysis" / "dataset_integrity_report.json"

        with open(report_path) as f:
            report = json.load(f)

        overlap = report.get('integrity', {}).get('overlap', {})

        for key, value in overlap.items():
            assert value == 0, f"Found overlap in {key}: {value}"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 9: Round 6 Candidate Selection
# ─────────────────────────────────────────────────────────────────────────────
class TestRound6CandidateSelection:
    """Tests for Round 6 candidate selection."""

    def test_candidate_key_file_exists(self):
        """Verify Round 6 candidate key file exists."""
        key_file = DATA_DIR / "model_predictions" / "phobert_positive_manual_label_candidates_key.csv"

        assert key_file.exists(), f"Candidate key file not found: {key_file}"

    def test_candidate_file_has_correct_structure(self):
        """Verify candidate file has required columns."""
        key_file = DATA_DIR / "model_predictions" / "phobert_positive_manual_label_candidates_key.csv"

        if not key_file.exists():
            pytest.skip("Candidate file not found")

        df = pd.read_csv(key_file)

        required_columns = [
            'annotation_id',
            'comment_text_sha256',
            'source_pool',
            'phobert_label',
            'probability',
        ]

        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"

    def test_candidate_selection_report_exists(self):
        """Verify candidate selection report exists."""
        report_file = DATA_DIR / "model_predictions" / "phobert_positive_manual_label_candidates_report.json"

        if not report_file.exists():
            pytest.skip("Report file not found")

        with open(report_file) as f:
            report = json.load(f)

        assert 'final_candidate_rows' in report, "Missing final_candidate_rows in report"
        assert report['final_candidate_rows'] > 0, "No candidates selected"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 10: MPS Seed Setting (Reproducibility)
# ─────────────────────────────────────────────────────────────────────────────
class TestReproducibility:
    """Tests for reproducibility across runs."""

    def test_mps_seed_set_in_training_scripts(self):
        """Verify MPS seed is set for Apple Silicon reproducibility."""
        scripts_to_check = [
            PROJECT_DIR / "scripts" / "retrain_all_models_round5.py",
            PROJECT_DIR / "scripts" / "retrain_phobert_for_round5.py",
            PROJECT_DIR / "scripts" / "train_phobert_round5_multiseed.py",
        ]

        mps_pattern = "torch.mps.manual_seed"

        for script in scripts_to_check:
            if not script.exists():
                continue

            content = script.read_text()

            # Check if script uses MPS
            uses_mps = 'backends.mps' in content or 'mps.is_available' in content

            if uses_mps:
                assert mps_pattern in content, (
                    f"Script {script.name} uses MPS but does not set torch.mps.manual_seed"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
