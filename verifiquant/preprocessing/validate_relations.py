from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _is_global_v2_rule(rule: Dict[str, Any]) -> bool:
    rid = str(rule.get("rule_id", "")).strip().lower()
    return rid.startswith("global_")


def _collect_core_maps(core_cards: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, str], Dict[str, Any]], List[str]]:
    core_by_id: Dict[str, Dict[str, Any]] = {}
    rules_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    errors: List[str] = []

    for idx, core in enumerate(core_cards, start=1):
        fic_id = str(core.get("fic_id", "")).strip()
        if not fic_id:
            errors.append(f"core[{idx}] missing fic_id")
            continue
        if fic_id in core_by_id:
            errors.append(f"duplicate core fic_id: {fic_id}")
            continue
        core_by_id[fic_id] = core

        checks = core.get("diagnostic_checks", [])
        if not isinstance(checks, list) or not checks:
            errors.append(f"core[{fic_id}] missing diagnostic_checks")
            continue

        local_rule_ids = set()
        for chk in checks:
            rid = str(chk.get("rule_id", "")).strip()
            if not rid:
                errors.append(f"core[{fic_id}] check missing rule_id")
                continue
            if rid in local_rule_ids:
                errors.append(f"core[{fic_id}] duplicate rule_id: {rid}")
                continue
            local_rule_ids.add(rid)

            d_type = str(chk.get("diagnostic_type", "")).strip().upper()
            if d_type not in {"F", "E"}:
                errors.append(f"core[{fic_id}] rule {rid} has invalid diagnostic_type={d_type} (core must be F/E only)")
            c_type = str(chk.get("check_type", "")).strip().lower()
            p_mode = str(chk.get("predicate_mode", "")).strip().lower()
            if c_type in {"deterministic", "normalization"} and p_mode and p_mode not in {"violation", "validity"}:
                errors.append(
                    f"core[{fic_id}] rule {rid} has invalid predicate_mode={p_mode} for {c_type} check"
                )

            rules_by_key[(fic_id, rid)] = chk

    return core_by_id, rules_by_key, errors


def validate_artifact_relations(
    *,
    core_cards: List[Dict[str, Any]],
    retrieval_cards: List[Dict[str, Any]],
    repair_rules: List[Dict[str, Any]],
) -> Dict[str, Any]:
    core_by_id, rules_by_key, errors = _collect_core_maps(core_cards)

    retrieval_by_fic: Dict[str, Dict[str, Any]] = {}
    for idx, r in enumerate(retrieval_cards, start=1):
        fic_id = str(r.get("fic_id", "")).strip()
        if not fic_id:
            errors.append(f"retrieval[{idx}] missing fic_id")
            continue
        if fic_id in retrieval_by_fic:
            errors.append(f"duplicate retrieval fic_id: {fic_id}")
            continue
        retrieval_by_fic[fic_id] = r

        core = core_by_id.get(fic_id)
        if not core:
            errors.append(f"retrieval[{fic_id}] has no matching core")
            continue
        if str(r.get("domain", "")).strip() != str(core.get("domain", "")).strip():
            errors.append(f"retrieval[{fic_id}] domain mismatch with core")
        if str(r.get("topic", "")).strip() != str(core.get("topic", "")).strip():
            errors.append(f"retrieval[{fic_id}] topic mismatch with core")

    for fic_id in core_by_id:
        if fic_id not in retrieval_by_fic:
            errors.append(f"core[{fic_id}] missing retrieval card")

    repair_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for idx, rr in enumerate(repair_rules, start=1):
        fic_id = str(rr.get("fic_id", "")).strip()
        rid = str(rr.get("rule_id", "")).strip()
        if not fic_id or not rid:
            errors.append(f"repair[{idx}] missing fic_id/rule_id")
            continue
        key = (fic_id, rid)
        if key in repair_by_key:
            errors.append(f"duplicate repair rule for fic_id={fic_id}, rule_id={rid}")
            continue
        repair_by_key[key] = rr

        if _is_global_v2_rule(rr):
            d_repair = str(rr.get("diagnostic_type", "")).strip().upper()
            if d_repair not in {"N", "I", "I_HARD", "I_SOFT"}:
                errors.append(
                    f"global repair {fic_id}/{rid} must use diagnostic_type N, I, I_HARD, or I_SOFT, got {d_repair}"
                )
            continue

        d_repair = str(rr.get("diagnostic_type", "")).strip().upper()
        # I-class rules are semantic guides generated from semantic_hints and
        # do not require 1:1 mapping to core F/E diagnostic_checks.
        if d_repair in ("I", "I_HARD", "I_SOFT"):
            continue

        if key not in rules_by_key:
            errors.append(f"repair rule {fic_id}/{rid} has no matching core diagnostic_check")
            continue

        core_check = rules_by_key[key]
        d_core = str(core_check.get("diagnostic_type", "")).strip().upper()
        if d_repair != d_core:
            errors.append(f"repair {fic_id}/{rid} diagnostic_type mismatch: core={d_core}, repair={d_repair}")

    for key in rules_by_key:
        if key not in repair_by_key:
            errors.append(f"missing repair rule for core diagnostic_check {key[0]}/{key[1]}")

    if errors:
        preview = "\n".join(f"- {x}" for x in errors[:20])
        suffix = "\n- ..." if len(errors) > 20 else ""
        raise ValueError(
            "Artifact relation validation failed:\n"
            f"{preview}{suffix}"
        )

    return {
        "core_count": len(core_cards),
        "retrieval_count": len(retrieval_cards),
        "repair_count": len(repair_rules),
        "diagnostic_rule_count": len(rules_by_key),
    }
