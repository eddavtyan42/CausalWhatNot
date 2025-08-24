from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, List

import gzip
import pandas as pd
import networkx as nx
import numpy as np
import logging

# Predefined edges and state counts for benchmark networks to avoid shipping
# large BIF files. These were extracted from the original network definitions.

# Canonical 8-node Asia network
ASIA_EDGES: list[tuple[str, str]] = [
    ("VisitAsia", "Tuberculosis"),
    ("Smoking", "LungCancer"),
    ("Smoking", "Bronchitis"),
    ("Tuberculosis", "Either"),
    ("LungCancer", "Either"),
    ("Either", "Xray"),
    ("Either", "Dyspnea"),
    ("Bronchitis", "Dyspnea"),
]

ASIA_STATES: dict[str, int] = {
    "VisitAsia": 2,
    "Tuberculosis": 2,
    "Smoking": 2,
    "LungCancer": 2,
    "Bronchitis": 2,
    "Either": 2,
    "Xray": 2,
    "Dyspnea": 2,
}

SACHS_EDGES: list[tuple[str, str]] = [
    ("Erk", "Akt"),
    ("PKA", "Akt"),
    ("Mek", "Erk"),
    ("PKA", "Erk"),
    ("PKA", "Jnk"),
    ("PKC", "Jnk"),
    ("PKA", "Mek"),
    ("PKC", "Mek"),
    ("Raf", "Mek"),
    ("PKA", "P38"),
    ("PKC", "P38"),
    ("PIP3", "PIP2"),
    ("Plcg", "PIP2"),
    ("Plcg", "PIP3"),
    ("PKC", "PKA"),
    ("PKA", "Raf"),
    ("PKC", "Raf"),
]

ALARM_EDGES: list[tuple[str, str]] = [
    ("LVFAILURE", "HISTORY"),
    ("LVEDVOLUME", "CVP"),
    ("LVEDVOLUME", "PCWP"),
    ("HYPOVOLEMIA", "LVEDVOLUME"),
    ("LVFAILURE", "LVEDVOLUME"),
    ("HYPOVOLEMIA", "STROKEVOLUME"),
    ("LVFAILURE", "STROKEVOLUME"),
    ("ERRLOWOUTPUT", "HRBP"),
    ("HR", "HRBP"),
    ("ERRCAUTER", "HREKG"),
    ("HR", "HREKG"),
    ("ERRCAUTER", "HRSAT"),
    ("HR", "HRSAT"),
    ("ANAPHYLAXIS", "TPR"),
    ("ARTCO2", "EXPCO2"),
    ("VENTLUNG", "EXPCO2"),
    ("INTUBATION", "MINVOL"),
    ("VENTLUNG", "MINVOL"),
    ("FIO2", "PVSAT"),
    ("VENTALV", "PVSAT"),
    ("PVSAT", "SAO2"),
    ("SHUNT", "SAO2"),
    ("PULMEMBOLUS", "PAP"),
    ("INTUBATION", "SHUNT"),
    ("PULMEMBOLUS", "SHUNT"),
    ("INTUBATION", "PRESS"),
    ("KINKEDTUBE", "PRESS"),
    ("VENTTUBE", "PRESS"),
    ("MINVOLSET", "VENTMACH"),
    ("DISCONNECT", "VENTTUBE"),
    ("VENTMACH", "VENTTUBE"),
    ("INTUBATION", "VENTLUNG"),
    ("KINKEDTUBE", "VENTLUNG"),
    ("VENTTUBE", "VENTLUNG"),
    ("INTUBATION", "VENTALV"),
    ("VENTLUNG", "VENTALV"),
    ("VENTALV", "ARTCO2"),
    ("ARTCO2", "CATECHOL"),
    ("INSUFFANESTH", "CATECHOL"),
    ("SAO2", "CATECHOL"),
    ("TPR", "CATECHOL"),
    ("CATECHOL", "HR"),
    ("HR", "CO"),
    ("STROKEVOLUME", "CO"),
    ("CO", "BP"),
    ("TPR", "BP"),
]

ALARM_STATES: dict[str, int] = {
    "HISTORY": 2,
    "CVP": 3,
    "PCWP": 3,
    "HYPOVOLEMIA": 2,
    "LVEDVOLUME": 3,
    "LVFAILURE": 2,
    "STROKEVOLUME": 3,
    "ERRLOWOUTPUT": 2,
    "HRBP": 3,
    "HREKG": 3,
    "ERRCAUTER": 2,
    "HRSAT": 3,
    "INSUFFANESTH": 2,
    "ANAPHYLAXIS": 2,
    "TPR": 3,
    "EXPCO2": 4,
    "KINKEDTUBE": 2,
    "MINVOL": 4,
    "FIO2": 2,
    "PVSAT": 3,
    "SAO2": 3,
    "PAP": 3,
    "PULMEMBOLUS": 2,
    "SHUNT": 2,
    "INTUBATION": 3,
    "PRESS": 4,
    "DISCONNECT": 2,
    "MINVOLSET": 3,
    "VENTMACH": 4,
    "VENTTUBE": 4,
    "VENTLUNG": 4,
    "VENTALV": 4,
    "ARTCO2": 3,
    "CATECHOL": 2,
    "HR": 3,
    "CO": 3,
    "BP": 3,
}

CHILD_EDGES: list[tuple[str, str]] = [
    ("DuctFlow", "HypDistrib"),
    ("CardiacMixing", "HypDistrib"),
    ("CardiacMixing", "HypoxiaInO2"),
    ("LungParench", "HypoxiaInO2"),
    ("LungParench", "CO2"),
    ("LungParench", "ChestXray"),
    ("LungFlow", "ChestXray"),
    ("LungParench", "Grunting"),
    ("Sick", "Grunting"),
    ("LVH", "LVHreport"),
    ("HypDistrib", "LowerBodyO2"),
    ("HypoxiaInO2", "LowerBodyO2"),
    ("HypoxiaInO2", "RUQO2"),
    ("CO2", "CO2Report"),
    ("ChestXray", "XrayReport"),
    ("BirthAsphyxia", "Disease"),
    ("Grunting", "GruntingReport"),
    ("Disease", "Age"),
    ("Sick", "Age"),
    ("Disease", "LVH"),
    ("Disease", "DuctFlow"),
    ("Disease", "CardiacMixing"),
    ("Disease", "LungParench"),
    ("Disease", "LungFlow"),
    ("Disease", "Sick"),
]

CHILD_STATES: dict[str, int] = {
    "BirthAsphyxia": 2,
    "HypDistrib": 2,
    "HypoxiaInO2": 3,
    "CO2": 3,
    "ChestXray": 5,
    "Grunting": 2,
    "LVHreport": 2,
    "LowerBodyO2": 3,
    "RUQO2": 3,
    "CO2Report": 2,
    "XrayReport": 5,
    "Disease": 6,
    "GruntingReport": 2,
    "Age": 3,
    "LVH": 2,
    "DuctFlow": 3,
    "CardiacMixing": 4,
    "LungParench": 3,
    "LungFlow": 3,
    "Sick": 2,
}

# Insurance network with 27 variables
INSURANCE_EDGES: list[tuple[str, str]] = [
    ("SocioEcon", "GoodStudent"),
    ("Age", "GoodStudent"),
    ("Age", "SocioEcon"),
    ("Age", "RiskAversion"),
    ("SocioEcon", "RiskAversion"),
    ("SocioEcon", "VehicleYear"),
    ("RiskAversion", "VehicleYear"),
    ("Accident", "ThisCarDam"),
    ("RuggedAuto", "ThisCarDam"),
    ("MakeModel", "RuggedAuto"),
    ("VehicleYear", "RuggedAuto"),
    ("Antilock", "Accident"),
    ("Mileage", "Accident"),
    ("DrivQuality", "Accident"),
    ("SocioEcon", "MakeModel"),
    ("RiskAversion", "MakeModel"),
    ("DrivingSkill", "DrivQuality"),
    ("RiskAversion", "DrivQuality"),
    ("MakeModel", "Antilock"),
    ("VehicleYear", "Antilock"),
    ("Age", "DrivingSkill"),
    ("SeniorTrain", "DrivingSkill"),
    ("Age", "SeniorTrain"),
    ("RiskAversion", "SeniorTrain"),
    ("ThisCarDam", "ThisCarCost"),
    ("CarValue", "ThisCarCost"),
    ("Theft", "ThisCarCost"),
    ("AntiTheft", "Theft"),
    ("HomeBase", "Theft"),
    ("CarValue", "Theft"),
    ("MakeModel", "CarValue"),
    ("VehicleYear", "CarValue"),
    ("Mileage", "CarValue"),
    ("RiskAversion", "HomeBase"),
    ("SocioEcon", "HomeBase"),
    ("RiskAversion", "AntiTheft"),
    ("SocioEcon", "AntiTheft"),
    ("OtherCarCost", "PropCost"),
    ("ThisCarCost", "PropCost"),
    ("Accident", "OtherCarCost"),
    ("RuggedAuto", "OtherCarCost"),
    ("SocioEcon", "OtherCar"),
    ("Accident", "MedCost"),
    ("Age", "MedCost"),
    ("Cushioning", "MedCost"),
    ("RuggedAuto", "Cushioning"),
    ("Airbag", "Cushioning"),
    ("MakeModel", "Airbag"),
    ("VehicleYear", "Airbag"),
    ("Accident", "ILiCost"),
    ("DrivingSkill", "DrivHist"),
    ("RiskAversion", "DrivHist"),
]

INSURANCE_STATES: dict[str, int] = {
    "GoodStudent": 2,
    "Age": 3,
    "SocioEcon": 4,
    "RiskAversion": 4,
    "VehicleYear": 2,
    "ThisCarDam": 4,
    "RuggedAuto": 3,
    "Accident": 4,
    "MakeModel": 5,
    "DrivQuality": 3,
    "Mileage": 4,
    "Antilock": 2,
    "DrivingSkill": 3,
    "SeniorTrain": 2,
    "ThisCarCost": 4,
    "Theft": 2,
    "CarValue": 5,
    "HomeBase": 4,
    "AntiTheft": 2,
    "PropCost": 4,
    "OtherCarCost": 4,
    "OtherCar": 2,
    "MedCost": 4,
    "Cushioning": 4,
    "Airbag": 2,
    "ILiCost": 4,
    "DrivHist": 3,
}


BASE_DIR = Path(__file__).resolve().parents[1] / 'data'


def is_discrete(df: pd.DataFrame, max_unique: int = 20) -> bool:
    """Check if each column of ``df`` represents a discrete variable.

    A column is treated as discrete when it stores integral values
    (including booleans and categoricals) and the number of unique
    observed values does not exceed ``max_unique``. Float columns are
    allowed if all non-missing values are integer-like. The dataframe is
    considered discrete only when **all** columns satisfy these
    conditions.

    Parameters
    ----------
    df:
        Input dataframe.
    max_unique:
        Maximum number of distinct values permitted for a column to be
        regarded as discrete. Defaults to ``20`` which safely covers the
        benchmark datasets.
    """

    for col in df.columns:
        series = df[col]
        if (
            pd.api.types.is_integer_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        ):
            pass
        elif pd.api.types.is_float_dtype(series):
            vals = series.dropna()
            if not (vals == np.floor(vals)).all():
                return False
        else:
            return False

        if series.nunique(dropna=True) > max_unique:
            return False

    return True


def _sample_gaussian(G: nx.DiGraph, n: int, seed: int = 0) -> pd.DataFrame:
    logger = logging.getLogger("benchmark")
    logger.info("Sample gaussian: nodes=%d edges=%d n=%d seed=%d", G.number_of_nodes(), G.number_of_edges(), n, seed)
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(index=range(n))
    for node in nx.topological_sort(G):
        parents = list(G.predecessors(node))
        noise = rng.normal(size=n)
        if parents:
            w = rng.uniform(0.5, 1.5, size=len(parents))
            df[node] = df[parents].to_numpy().dot(w) + noise
        else:
            df[node] = noise
    return df


def _sample_discrete(
    G: nx.DiGraph, states: Dict[str, List[str] | int], n: int, seed: int = 0
) -> pd.DataFrame:
    """Sample discrete data by discretising Gaussian samples."""
    logger = logging.getLogger("benchmark")
    logger.info("Sample discrete: nodes=%d edges=%d n=%d seed=%d", G.number_of_nodes(), G.number_of_edges(), n, seed)
    cont = _sample_gaussian(G, n, seed)
    df = pd.DataFrame(index=range(n))
    for node in cont.columns:
        state_info = states.get(node, [0, 1])
        if isinstance(state_info, int):
            k = state_info
        else:
            k = len(state_info)
        quantiles = np.quantile(cont[node], np.linspace(0, 1, k + 1)[1:-1])
        df[node] = np.digitize(cont[node], quantiles)
    return df


def load_insurance(n_samples: int = 10000, force: bool = False) -> Tuple[pd.DataFrame, nx.DiGraph]:
    """Generate or load the Insurance benchmark dataset."""
    logger = logging.getLogger("benchmark")
    name = "insurance"
    data_dir = BASE_DIR / name
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / f"{name}_data.csv"

    G = nx.DiGraph()
    G.add_edges_from(INSURANCE_EDGES)
    states = INSURANCE_STATES

    if data_path.exists() and not force:
        df = pd.read_csv(data_path)
    else:
        df = _sample_discrete(G, states, n_samples)
        df.to_csv(data_path, index=False)
    logger.info("Loaded dataset '%s': shape=%s edges=%d path=%s", name, str(df.shape), G.number_of_edges(), str(data_path))

    return df, G

def load_dataset(name: str, n_samples: int = 10000, force: bool = False) -> Tuple[pd.DataFrame, nx.DiGraph]:
    """Load or generate samples for one of the benchmark datasets."""
    logger = logging.getLogger("benchmark")
    name = name.lower()
    data_dir = BASE_DIR / name
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / f"{name}_data.csv"

    if name == 'asia':
        G = nx.DiGraph()
        G.add_edges_from(ASIA_EDGES)
        states = ASIA_STATES
        if data_path.exists() and not force:
            df = pd.read_csv(data_path)
        else:
            df = _sample_discrete(G, states, n_samples)
            df.to_csv(data_path, index=False)
        logger.info("Loaded dataset '%s': shape=%s edges=%d path=%s", name, str(df.shape), G.number_of_edges(), str(data_path))
        return df, G

    elif name == "sachs":
        # Build graph from predefined edges
        G = nx.DiGraph()
        G.add_edges_from(SACHS_EDGES)
        if data_path.exists() and not force:
            df = pd.read_csv(data_path)
        else:
            df = _sample_gaussian(G, n_samples)
            df.to_csv(data_path, index=False)
        logger.info("Loaded dataset '%s': shape=%s edges=%d path=%s", name, str(df.shape), G.number_of_edges(), str(data_path))
        return df, G

    elif name == "alarm":
        G = nx.DiGraph()
        G.add_edges_from(ALARM_EDGES)
        states = ALARM_STATES
        if data_path.exists() and not force:
            df = pd.read_csv(data_path)
        else:
            df = _sample_discrete(G, states, n_samples)
            df.to_csv(data_path, index=False)
        logger.info("Loaded dataset '%s': shape=%s edges=%d path=%s", name, str(df.shape), G.number_of_edges(), str(data_path))
        return df, G

    elif name == "child":
        G = nx.DiGraph()
        G.add_edges_from(CHILD_EDGES)
        states = CHILD_STATES
        if data_path.exists() and not force:
            df = pd.read_csv(data_path)
        else:
            df = _sample_discrete(G, states, n_samples)
            df.to_csv(data_path, index=False)
        logger.info("Loaded dataset '%s': shape=%s edges=%d path=%s", name, str(df.shape), G.number_of_edges(), str(data_path))
        return df, G

    elif name == "insurance":
        logger.info("Loading dataset via helper: %s", name)
        return load_insurance(n_samples=n_samples, force=force)

    else:
        raise ValueError(f"Unknown dataset: {name}")
