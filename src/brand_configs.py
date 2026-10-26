import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.ACMPipelineClass import ACMPipeline


@dataclass(frozen=True)
class CommonPipelineConfig:
    """
    Shared ACM pipeline settings used by every brand.
    Keep only brand-specific values in per-brand JSON files.
    """

    algorithm_prefix: str
    instance_id: str
    membership_identifier: str
    amc_account_id: str
    use_case_type: str = "audience"
    base_path: str = "/home/ec2-user/SageMaker/ACM_Model_Creation/ml/code"
    metrics_name: str = "metrics_definition"
    cmaa_config_name: str = "config"
    schema_name: str = "schema"
    region: str = "us-east-1"
    verbose: bool = False


@dataclass(frozen=True)
class BrandConfig:
    """
    Values that are allowed to differ by brand.
    """

    key: str
    algorithm_suffix: str
    target_brands: List[str]
    fallback_brand_keywords: List[str]
    target_market: str
    brand_filter_list: str
    brand_tag: str

    @classmethod
    def from_dict(cls, key: str, payload: Dict[str, Any]) -> "BrandConfig":
        required_fields = [
            "algorithm_suffix",
            "target_brands",
            "fallback_brand_keywords",
            "target_market",
            "brand_filter_list",
            "brand_tag",
        ]
        missing = [field for field in required_fields if field not in payload]
        if missing:
            raise ValueError(
                f"Brand config '{key}' is missing required fields: {', '.join(missing)}"
            )

        return cls(
            key=key,
            algorithm_suffix=str(payload["algorithm_suffix"]),
            target_brands=[str(v) for v in payload["target_brands"]],
            fallback_brand_keywords=[str(v) for v in payload["fallback_brand_keywords"]],
            target_market=str(payload["target_market"]),
            brand_filter_list=str(payload["brand_filter_list"]),
            brand_tag=str(payload["brand_tag"]),
        )


def _default_brand_config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "API" / "brands"


def list_brand_keys(config_dir: Optional[Path] = None) -> List[str]:
    directory = Path(config_dir) if config_dir else _default_brand_config_dir()
    if not directory.exists():
        return []
    return sorted(path.stem for path in directory.glob("*.json"))


def load_brand_config(brand_key: str, config_dir: Optional[Path] = None) -> BrandConfig:
    directory = Path(config_dir) if config_dir else _default_brand_config_dir()
    config_path = directory / f"{brand_key}.json"
    if not config_path.exists():
        available = ", ".join(list_brand_keys(directory)) or "none"
        raise FileNotFoundError(
            f"Brand config not found for '{brand_key}' at {config_path}. "
            f"Available brand keys: {available}"
        )

    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Brand config '{brand_key}' must be a JSON object.")

    return BrandConfig.from_dict(brand_key, payload)


def build_algorithm_name(common: CommonPipelineConfig, brand: BrandConfig) -> str:
    return f"{common.algorithm_prefix}_{brand.algorithm_suffix}"


def create_pipeline_for_brand(
    brand_key: str,
    common: CommonPipelineConfig,
    cloud_formation_env_vars: Dict[str, Any],
    config_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Create a single ACMPipeline interface from shared + per-brand config.

    Returns:
        Dict[str, Any]: {"pipeline": ACMPipeline, "brand_config": BrandConfig, "algorithm_name": str}
    """

    brand = load_brand_config(brand_key, config_dir=config_dir)
    algorithm_name = build_algorithm_name(common, brand)

    pipeline = ACMPipeline(
        acm_algorithm_name=algorithm_name,
        instance_id=common.instance_id,
        membershipIdentifier=common.membership_identifier,
        amc_account_id=common.amc_account_id,
        use_case_type=common.use_case_type,
        base_path=common.base_path,
        metrics_name=common.metrics_name,
        cmaa_config_name=common.cmaa_config_name,
        schema_name=common.schema_name,
        region=common.region,
        verbose=common.verbose,
        **cloud_formation_env_vars,
    )

    return {
        "pipeline": pipeline,
        "brand_config": brand,
        "algorithm_name": algorithm_name,
    }
