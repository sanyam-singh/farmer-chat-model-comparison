import json
import time
import ast
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from queue import Queue
from typing import List, Dict, Any, Tuple

import numpy as np
import openai
import pandas as pd


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EnhancedFactEvaluatorProduction:
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key

    def check_valid_json(self, predicted_facts):
        """Fixed version of check_valid_json method"""
        try:
            if isinstance(predicted_facts, str):
                try:
                    # First try json.loads
                    parsed = json.loads(predicted_facts)
                except json.JSONDecodeError:
                    try:
                        # Then try ast.literal_eval for Python string representations
                        parsed = ast.literal_eval(predicted_facts)
                    except Exception:
                        return {
                            "is_valid": False,
                            "parsed_data": {"facts": []},
                            "error": f"Failed to parse string: {predicted_facts[:100]}...",
                        }
            elif isinstance(predicted_facts, list):
                parsed = predicted_facts
            elif isinstance(predicted_facts, dict):
                parsed = predicted_facts.get("facts", predicted_facts)
            else:
                return {
                    "is_valid": False,
                    "parsed_data": {"facts": []},
                    "error": f"Unsupported data type: {type(predicted_facts)}",
                }

            if isinstance(parsed, list):
                facts_list = parsed
            elif isinstance(parsed, dict) and "facts" in parsed:
                facts_list = parsed["facts"]
            else:
                facts_list = [parsed] if parsed else []

            return {"is_valid": True, "parsed_data": {"facts": facts_list}, "error": None}
        except Exception as e:
            return {"is_valid": False, "parsed_data": {"facts": []}, "error": str(e)}

    def extract_facts_by_category(self, facts_data: Any) -> Dict[str, List[Dict]]:
        """Extract facts grouped by category"""
        category_facts = defaultdict(list)

        if isinstance(facts_data, str):
            try:
                facts_data = json.loads(facts_data)
            except Exception:
                try:
                    facts_data = ast.literal_eval(facts_data)
                except Exception:
                    return category_facts

        if isinstance(facts_data, list):
            facts_list = facts_data
        elif isinstance(facts_data, dict) and "facts" in facts_data:
            facts_list = facts_data.get("facts", [])
        else:
            return category_facts

        for fact in facts_list:
            if isinstance(fact, dict):
                category = fact.get("category", "unknown")
                category_facts[category].append(fact)
            elif isinstance(fact, str):
                category_facts["unknown"].append({"fact": fact, "category": "unknown"})

        return dict(category_facts)

    def find_best_semantic_match(
        self, gold_fact: str, pred_facts: List[str], category: str, debug: bool = False
    ) -> Dict:
        """Find the best semantic match using the matching prompt"""

        if not pred_facts:
            return {"best_match": None, "reason": "No predicted facts available", "confidence": 0.0}

        matching_prompt = f"""You are an agricultural fact comparison expert. Compare the reference fact with the candidate facts to find the best semantic match based on agricultural meaning and context.

REFERENCE FACT (Category: {category}):
{gold_fact}

CANDIDATE FACTS:
{json.dumps(pred_facts, indent=2)}

INSTRUCTIONS:
1. Find the candidate fact that conveys the most similar agricultural meaning to the reference fact
2. Prioritize matches that share the same:
   - Crop/plant type
   - Agricultural practice or technique
   - Specific measurements, dosages, or timing
   - Expected outcomes or benefits
3. Consider facts as matching even with different wording if they convey equivalent agricultural advice
4. Focus on semantic similarity and practical agricultural application rather than exact word matching
5. If no candidate fact is semantically similar enough (confidence < 0.7), return null for best_match

RESPOND WITH ONLY JSON:
{{
    "best_match": "exact text of best matching candidate fact or null if no good match",
    "reason": "detailed explanation focusing on specific agricultural elements that align (crop type, practice, measurements, outcomes) or why no adequate match exists",
    "confidence": 0.0-1.0
}}
"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert agricultural fact comparison specialist. Respond ONLY with valid JSON.",
                },
                {"role": "user", "content": matching_prompt},
            ]

            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0,
                max_tokens=1000,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content.strip()
            data = json.loads(content)
            best_match = data.get("best_match")
            reason = data.get("reason", "No reason provided")
            confidence = data.get("confidence", 0.0)

            if best_match and best_match in pred_facts:
                return {"best_match": best_match, "reason": reason, "confidence": confidence}
            return {"best_match": None, "reason": reason, "confidence": 0.0}

        except Exception as e:
            if debug:
                logger.error(f"Error in finding best match: {e}")
            return {"best_match": None, "reason": f"Error during matching: {str(e)}", "confidence": 0.0}

    def check_contradictions(
        self, gold_fact: str, pred_facts: List[str], category: str, debug: bool = False
    ) -> List[Dict]:
        """Check for contradictions between gold fact and predicted facts"""

        if not pred_facts:
            return []

        contradiction_prompt = f"""
You are an agricultural contradiction-detection expert. Your task: IDENTIFY ONLY genuine contradictions between a single REFERENCE FACT and a list of CANDIDATE FACTS, and EXPLAIN each finding with a short, structured justification (NOT internal chain-of-thought).

REFERENCE FACT (Category: {category}):
{gold_fact}

CANDIDATE FACTS:
{json.dumps(pred_facts, indent=2)}

--- INSTRUCTIONS & OVERVIEW ---
1) Output: ONLY a single JSON object (see schema below). Do NOT produce any text outside JSON.
2) Do NOT reveal internal chain-of-thought. Instead provide a concise, structured summary of the evaluation steps used for each contradiction (max 2–3 short sentences / bullet-like items).
3) A *genuine contradiction* = two facts that make OPPOSITE or CONFLICTING claims about the SAME agricultural aspect (same subject and same property/attribute). Consider compound statements component-wise (temperature, humidity, timing, quantity, effect, method, scale, nutrient, crop, season, or location).

... (prompt truncated for brevity in code comments) ...
"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert agricultural contradiction detection specialist. Respond ONLY with valid JSON.",
                },
                {"role": "user", "content": contradiction_prompt},
            ]

            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content.strip()
            data = json.loads(content)
            return data.get("contradictions", [])

        except Exception as e:
            if debug:
                logger.error(f"Error in contradiction detection: {e}")
            return []

    def evaluate_unmatched_relevance(
        self, question: str, ground_facts: List[str], unmatched_facts: List[str], debug: bool = False
    ) -> Dict:
        """Evaluate unmatched facts for relevance and quality"""

        if not unmatched_facts:
            return {
                "relevant_facts": [],
                "irrelevant_facts": [],
                "analysis_results": [],
            }

        unmatched_relevant_prompt = f"""You are an agricultural expert tasked with analyzing the relevance and accuracy of predicted facts in relation to specific agricultural questions and ground truth facts. Your goal is to evaluate how well each predicted fact addresses the given question, aligns with established ground facts, and determine its practical value for farmers.

... (prompt truncated for brevity in code comments) ...

QUESTION: {question}
GROUND_FACTS: {json.dumps(ground_facts)}
PREDICTED_FACTS: {json.dumps(unmatched_facts)}
"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert agricultural fact evaluation specialist. Respond ONLY with valid JSON.",
                },
                {"role": "user", "content": unmatched_relevant_prompt},
            ]

            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0,
                max_tokens=3000,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content.strip()
            data = json.loads(content)

            relevant_facts = []
            irrelevant_facts = []

            for analysis in data.get("predicted_facts_analysis", []):
                if analysis.get("overall_score", 0) >= 6:
                    relevant_facts.append(
                        {
                            "fact": analysis["predicted_fact"],
                            "score": analysis["overall_score"],
                            "reason": analysis["explanation"],
                        }
                    )
                else:
                    irrelevant_facts.append(
                        {
                            "fact": analysis["predicted_fact"],
                            "score": analysis["overall_score"],
                            "reason": analysis["explanation"],
                        }
                    )

            return {
                "relevant_facts": relevant_facts,
                "irrelevant_facts": irrelevant_facts,
                "analysis_results": data.get("predicted_facts_analysis", []),
                "summary": data.get("summary", {}),
            }

        except Exception as e:
            if debug:
                logger.error(f"Error in unmatched relevance evaluation: {e}")
            return {
                "relevant_facts": [],
                "irrelevant_facts": unmatched_facts,
                "analysis_results": [],
                "error": str(e),
            }

    def enhanced_category_wise_matching(
        self, predicted_facts: Any, golden_facts: Any, question: str = "", debug: bool = False
    ) -> Dict[str, Any]:
        """Enhanced matching with proper error handling"""

        pred_by_category = self.extract_facts_by_category(predicted_facts)
        gold_by_category = self.extract_facts_by_category(golden_facts)

        if debug:
            logger.info(f"Predicted categories: {list(pred_by_category.keys())}")
            logger.info(f"Golden categories: {list(gold_by_category.keys())}")

        all_results = []
        total_matches = 0
        total_gold_facts = 0
        total_contradictions = 0
        total_relevant_unmatched = 0
        total_irrelevant_unmatched = 0

        for category in gold_by_category.keys():
            gold_facts_in_category = gold_by_category[category]
            pred_facts_in_category = pred_by_category.get(category, [])

            total_gold_facts += len(gold_facts_in_category)

            gold_fact_texts = [self._extract_fact_text(fact) for fact in gold_facts_in_category]
            pred_fact_texts = [self._extract_fact_text(fact) for fact in pred_facts_in_category]

            used_pred_facts = set()
            category_results = []

            for gold_fact in gold_fact_texts:
                available_pred_facts = [f for f in pred_fact_texts if f not in used_pred_facts]

                if not available_pred_facts:
                    category_results.append(
                        {
                            "gold_fact": gold_fact,
                            "matched_pred_fact": None,
                            "match_reason": "No available predicted facts",
                            "match_confidence": 0.0,
                            "status": "unmatched_gold",
                        }
                    )
                    continue

                match_result = self.find_best_semantic_match(
                    gold_fact, available_pred_facts, category, debug=debug
                )

                if match_result["best_match"] and match_result["confidence"] >= 0.7:
                    used_pred_facts.add(match_result["best_match"])
                    total_matches += 1

                    category_results.append(
                        {
                            "gold_fact": gold_fact,
                            "matched_pred_fact": match_result["best_match"],
                            "match_reason": match_result["reason"],
                            "match_confidence": match_result["confidence"],
                            "status": "matched",
                        }
                    )
                else:
                    category_results.append(
                        {
                            "gold_fact": gold_fact,
                            "matched_pred_fact": None,
                            "match_reason": match_result["reason"],
                            "match_confidence": match_result["confidence"],
                            "status": "unmatched_gold",
                        }
                    )

            leftover_pred_facts = [f for f in pred_fact_texts if f not in used_pred_facts]

            if leftover_pred_facts:
                for pred_fact in leftover_pred_facts:
                    contradictions = self.check_contradictions(
                        pred_fact, gold_fact_texts, category, debug=debug
                    )

                    if contradictions:
                        total_contradictions += len(contradictions)
                        category_results.append(
                            {
                                "pred_fact": pred_fact,
                                "status": "contradictory",
                                "contradictions": contradictions,
                                "contradiction_count": len(contradictions),
                            }
                        )
                    else:
                        try:
                            relevance_result = self.evaluate_unmatched_relevance(
                                question, gold_fact_texts, [pred_fact], debug=debug
                            )

                            if "error" in relevance_result or not isinstance(relevance_result, dict):
                                total_irrelevant_unmatched += 1
                                category_results.append(
                                    {
                                        "pred_fact": pred_fact,
                                        "status": "unmatched_irrelevant",
                                        "relevance_analysis": {"error": str(relevance_result)},
                                        "relevance_score": 0,
                                    }
                                )
                            elif relevance_result.get("relevant_facts"):
                                total_relevant_unmatched += 1
                                relevant_fact = relevance_result["relevant_facts"][0]
                                category_results.append(
                                    {
                                        "pred_fact": pred_fact,
                                        "status": "unmatched_relevant",
                                        "relevance_analysis": relevance_result["analysis_results"][0]
                                        if relevance_result.get("analysis_results")
                                        else {},
                                        "relevance_score": relevant_fact.get("score", 0)
                                        if isinstance(relevant_fact, dict)
                                        else 0,
                                    }
                                )
                            else:
                                total_irrelevant_unmatched += 1
                                irrelevant_facts = relevance_result.get("irrelevant_facts", [])
                                relevance_score = 0
                                if irrelevant_facts and isinstance(irrelevant_facts, list):
                                    first = irrelevant_facts[0]
                                    if isinstance(first, dict):
                                        relevance_score = first.get("score", 0)

                                category_results.append(
                                    {
                                        "pred_fact": pred_fact,
                                        "status": "unmatched_irrelevant",
                                        "relevance_analysis": relevance_result["analysis_results"][0]
                                        if relevance_result.get("analysis_results")
                                        else {},
                                        "relevance_score": relevance_score,
                                    }
                                )
                        except Exception as e:
                            logger.error(f"Error in relevance evaluation: {e}")
                            total_irrelevant_unmatched += 1
                            category_results.append(
                                {
                                    "pred_fact": pred_fact,
                                    "status": "unmatched_irrelevant",
                                    "relevance_analysis": {"error": str(e)},
                                    "relevance_score": 0,
                                }
                            )

            all_results.extend(category_results)

        iogt_score = total_matches / total_gold_facts if total_gold_facts > 0 else 0

        return {
            "iogt_score": iogt_score,
            "total_matches": total_matches,
            "total_gold_facts": total_gold_facts,
            "total_contradictions": total_contradictions,
            "total_relevant_unmatched": total_relevant_unmatched,
            "total_irrelevant_unmatched": total_irrelevant_unmatched,
            "detailed_results": all_results,
            "summary_stats": {
                "match_rate": iogt_score,
                "contradiction_rate": total_contradictions
                / len([r for r in all_results if "pred_fact" in r])
                if any("pred_fact" in r for r in all_results)
                else 0,
                "relevant_unmatched_rate": total_relevant_unmatched
                / len([r for r in all_results if "pred_fact" in r])
                if any("pred_fact" in r for r in all_results)
                else 0,
                "irrelevant_rate": total_irrelevant_unmatched
                / len([r for r in all_results if "pred_fact" in r])
                if any("pred_fact" in r for r in all_results)
                else 0,
            },
        }

    def _extract_fact_text(self, fact_item: Any) -> str:
        """Extract fact text from various formats"""
        if isinstance(fact_item, str):
            return fact_item
        if isinstance(fact_item, dict):
            return fact_item.get("fact", str(fact_item))
        return str(fact_item)

    def evaluate_fact_sft_model_enhanced(
        self, predicted_facts: Any, golden_facts: Any, question: str = "", debug: bool = False
    ) -> Dict[str, Any]:
        """Enhanced evaluation with new strategy"""

        metrics = {}

        json_result = self.check_valid_json(predicted_facts)
        metrics["json_validity"] = json_result["is_valid"]

        pred_data = json_result["parsed_data"] if json_result["is_valid"] else predicted_facts

        metrics["enhanced_matching"] = self.enhanced_category_wise_matching(
            pred_data, golden_facts, question, debug=debug
        )

        return metrics

    def export_results_to_csv(self, results: List[Dict], filename: str = "enhanced_evaluation_results.csv"):
        """Export detailed results to CSV format"""

        csv_rows = []

        for result in results:
            base_info = {
                "question": result.get("question", ""),
                "sample_id": result.get("sample_id", ""),
                "iogt_score": result.get("iogt_score", 0),
                "total_matches": result.get("total_matches", 0),
                "total_gold_facts": result.get("total_gold_facts", 0),
                "total_contradictions": result.get("total_contradictions", 0),
                "total_relevant_unmatched": result.get("total_relevant_unmatched", 0),
                "total_irrelevant_unmatched": result.get("total_irrelevant_unmatched", 0),
            }

            for detail in result.get("detailed_results", []):
                row = base_info.copy()

                if detail.get("status") == "matched":
                    row.update(
                        {
                            "fact_type": "gold_fact",
                            "fact_text": detail.get("gold_fact", ""),
                            "matched_fact": detail.get("matched_pred_fact", ""),
                            "status": "matched",
                            "confidence": detail.get("match_confidence", 0),
                            "reason": detail.get("match_reason", ""),
                            "contradiction_count": 0,
                            "relevance_score": 10,
                        }
                    )
                elif detail.get("status") == "contradictory":
                    contradictions_json = json.dumps(detail.get("contradictions", []), ensure_ascii=False)
                    row.update(
                        {
                            "fact_type": "pred_fact",
                            "fact_text": detail.get("pred_fact", ""),
                            "matched_fact": "",
                            "status": "contradictory",
                            "confidence": 0,
                            "reason": f"Contradicts {detail.get('contradiction_count', 0)} gold facts",
                            "contradiction_count": detail.get("contradiction_count", 0),
                            "relevance_score": 0,
                            "contradictions_full": contradictions_json,
                        }
                    )
                elif detail.get("status") == "unmatched_relevant":
                    row.update(
                        {
                            "fact_type": "pred_fact",
                            "fact_text": detail.get("pred_fact", ""),
                            "matched_fact": "",
                            "status": "unmatched_relevant",
                            "confidence": 0,
                            "reason": detail.get("relevance_analysis", {}).get("explanation", ""),
                            "contradiction_count": 0,
                            "relevance_score": detail.get("relevance_score", 0),
                        }
                    )
                elif detail.get("status") == "unmatched_irrelevant":
                    row.update(
                        {
                            "fact_type": "pred_fact",
                            "fact_text": detail.get("pred_fact", ""),
                            "matched_fact": "",
                            "status": "unmatched_irrelevant",
                            "confidence": 0,
                            "reason": detail.get("relevance_analysis", {}).get("explanation", ""),
                            "contradiction_count": 0,
                            "relevance_score": detail.get("relevance_score", 0),
                        }
                    )
                elif detail.get("status") == "unmatched_gold":
                    row.update(
                        {
                            "fact_type": "gold_fact",
                            "fact_text": detail.get("gold_fact", ""),
                            "matched_fact": "",
                            "status": "unmatched_gold",
                            "confidence": 0,
                            "reason": detail.get("match_reason", ""),
                            "contradiction_count": 0,
                            "relevance_score": 0,
                        }
                    )

                csv_rows.append(row)

        df = pd.DataFrame(csv_rows)
        df.to_csv(filename, index=False)
        logger.info(f"Results exported to {filename}")

        return df


def process_single_row(args):
    """Process a single row with proper parsing"""
    idx, row_data, openai_api_key = args

    try:
        evaluator = EnhancedFactEvaluatorProduction(openai_api_key)

        predicted_facts_raw = row_data["FT_response"]
        golden_facts_raw = row_data["gt_facts"]
        question = row_data.get("question", "")

        def parse_if_string(value):
            if isinstance(value, str):
                try:
                    return ast.literal_eval(value)
                except Exception:
                    return []
            return value if isinstance(value, (list, dict)) else []

        predicted_facts = parse_if_string(predicted_facts_raw)
        golden_facts = parse_if_string(golden_facts_raw)

        metrics = evaluator.evaluate_fact_sft_model_enhanced(
            predicted_facts, golden_facts, question, debug=False
        )

        result = {
            "sample_id": idx,
            "question": question,
            "predicted_facts": predicted_facts,
            "golden_facts": golden_facts,
            "json_validity": metrics["json_validity"],
            **metrics["enhanced_matching"],
        }

        return result

    except Exception as e:
        logger.error(f"Error processing row {idx}: {e}")
        return {
            "sample_id": idx,
            "question": row_data.get("question", ""),
            "predicted_facts": row_data.get("FT_response", ""),
            "golden_facts": row_data.get("gt_facts", ""),
            "error": str(e),
        }


class ProgressTracker:
    def __init__(self, total_tasks):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.lock = None
        self.start_time = time.time()

    def update(self, increment: int = 1):
        self.completed_tasks += increment
        if self.completed_tasks % 10 == 0 or self.completed_tasks == self.total_tasks:
            elapsed_time = time.time() - self.start_time
            rate = self.completed_tasks / elapsed_time if elapsed_time > 0 else 0
            eta = (self.total_tasks - self.completed_tasks) / rate if rate > 0 else 0

            logger.info(
                f"Progress: {self.completed_tasks}/{self.total_tasks} "
                f"({self.completed_tasks/self.total_tasks*100:.1f}%) "
                f"Rate: {rate:.2f} samples/sec "
                f"ETA: {eta/60:.1f} minutes"
            )


def integrate_enhanced_evaluation_v2_threaded(
    df: pd.DataFrame, openai_api_key: str = None, max_workers: int = 40
) -> List[Dict]:
    """Threaded implementation for I/O bound tasks (API calls)"""
    logger.info(
        f"Starting threaded enhanced evaluation with {max_workers} workers for {len(df)} samples..."
    )

    args_list = [(idx, row.to_dict(), openai_api_key) for idx, row in df.iterrows()]

    progress_tracker = ProgressTracker(len(args_list))
    results: List[Dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(process_single_row, args): args[0] for args in args_list}

        for future in as_completed(future_to_idx):
            try:
                result = future.result()
                results.append(result)
                progress_tracker.update()
            except Exception as e:
                idx = future_to_idx[future]
                logger.error(f"Task {idx} generated an exception: {e}")
                results.append({"sample_id": idx, "error": str(e)})
                progress_tracker.update()

    results.sort(key=lambda x: x.get("sample_id", 0))

    logger.info(f"Completed processing {len(results)} samples")
    return results


def run_enhanced_evaluation_pipeline_optimized(
    df: pd.DataFrame,
    openai_api_key: str,
    output_filename: str = "enhanced_evaluation_results.csv",
    max_workers: int = 40,
) -> Tuple[pd.DataFrame, Dict]:
    """Complete optimized pipeline for enhanced evaluation with CSV export"""
    start_time = time.time()

    results = integrate_enhanced_evaluation_v2_threaded(df, openai_api_key, max_workers)

    evaluator = EnhancedFactEvaluatorProduction(openai_api_key)
    results_df = evaluator.export_results_to_csv(results, output_filename)

    valid_results = [r for r in results if "error" not in r]
    summary_stats = {
        "total_samples": len(results),
        "successful_samples": len(valid_results),
        "failed_samples": len(results) - len(valid_results),
        "avg_iogt_score": np.mean([r.get("iogt_score", 0) for r in valid_results]) if valid_results else 0,
        "avg_matches": np.mean([r.get("total_matches", 0) for r in valid_results]) if valid_results else 0,
        "avg_gold_facts": np.mean([r.get("total_gold_facts", 0) for r in valid_results])
        if valid_results
        else 0,
        "avg_contradictions": np.mean([r.get("total_contradictions", 0) for r in valid_results])
        if valid_results
        else 0,
        "avg_relevant_unmatched": np.mean(
            [r.get("total_relevant_unmatched", 0) for r in valid_results]
        )
        if valid_results
        else 0,
        "avg_irrelevant_unmatched": np.mean(
            [r.get("total_irrelevant_unmatched", 0) for r in valid_results]
        )
        if valid_results
        else 0,
        "processing_time_minutes": (time.time() - start_time) / 60,
        "samples_per_minute": len(results) / ((time.time() - start_time) / 60),
    }

    logger.info("\n" + "=" * 80)
    logger.info("           OPTIMIZED ENHANCED EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Samples: {summary_stats['total_samples']}")
    logger.info(f"Successful Samples: {summary_stats['successful_samples']}")
    logger.info(f"Failed Samples: {summary_stats['failed_samples']}")
    logger.info(f"Average IoGT Score: {summary_stats['avg_iogt_score']:.3f}")
    logger.info(f"Average Matches per Sample: {summary_stats['avg_matches']:.1f}")
    logger.info(f"Average Gold Facts per Sample: {summary_stats['avg_gold_facts']:.1f}")
    logger.info(f"Average Contradictions per Sample: {summary_stats['avg_contradictions']:.1f}")
    logger.info(
        f"Average Relevant Unmatched per Sample: {summary_stats['avg_relevant_unmatched']:.1f}"
    )
    logger.info(
        f"Average Irrelevant Unmatched per Sample: {summary_stats['avg_irrelevant_unmatched']:.1f}"
    )
    logger.info(f"Processing Time: {summary_stats['processing_time_minutes']:.1f} minutes")
    logger.info(f"Processing Rate: {summary_stats['samples_per_minute']:.1f} samples/minute")
    logger.info(f"Workers Used: {max_workers}")
    logger.info(f"\nDetailed results saved to: {output_filename}")

    return results_df, summary_stats

