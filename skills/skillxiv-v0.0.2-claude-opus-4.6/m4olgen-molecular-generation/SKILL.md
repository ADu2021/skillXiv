---
name: m4olgen-molecular-generation
title: "M4olGen: Multi-Agent, Multi-Stage Molecular Generation under Precise Multi-Property Constraints"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.10131"
keywords: [molecular-generation, multi-agent, property-constraints, fragment-based, GRPO]
description: "Generates molecules meeting precise numeric property constraints across multiple dimensions through two-stage multi-agent framework with fragment-level edits and Group Relative Policy Optimization, improving validity and property satisfaction."
---

## Overview

Design a multi-agent molecular generation system that creates valid molecules satisfying precise numeric constraints on multiple physicochemical properties (QED, LogP, Molecular Weight, HOMO, LUMO). Use a two-stage approach with fragment-level reasoning to enable controlled, property-aware generation.

## When to Use

- For drug discovery requiring molecules meeting specific property constraints
- When you need precise multi-property control (not just single objectives)
- For molecular optimization with numeric property bounds
- When working with structure-based design constraints

## When NOT to Use

- For property prediction or molecular analysis
- When single-property optimization is sufficient
- For molecules where fragment-based reasoning doesn't apply
- In environments lacking pretrained molecular models

## Key Technical Components

### Stage I: Prototype Generation via Multi-Agent Fragment Editing

Generate candidate molecules through multi-agent collaboration using fragment-level operations.

```python
# Multi-agent prototype generation
class PrototypeGenerator:
    def __init__(self):
        self.fragment_retriever = FragmentRetriever()
        self.editor_agent = FragmentEditorAgent()
        self.validator_agent = StructuralValidatorAgent()

    def generate_prototype(self, property_constraints, num_candidates=10):
        """Generate candidate molecules near feasible region"""
        candidates = []

        for _ in range(num_candidates):
            # Start with seed molecule
            molecule = self.select_seed_molecule(property_constraints)

            # Multi-agent editing
            current_molecule = molecule
            for edit_step in range(MAX_EDITS):
                # Fragment-level edit
                edit_proposal = self.editor_agent.propose_edit(
                    current_molecule,
                    property_constraints
                )

                # Retrieve relevant fragments
                fragments = self.fragment_retriever.retrieve(
                    edit_proposal["target_fragment"],
                    k=5
                )

                # Apply edit using retrieved fragments
                for candidate_fragment in fragments:
                    edited = self.apply_fragment_edit(
                        current_molecule,
                        edit_proposal["position"],
                        candidate_fragment
                    )

                    # Validate structure
                    if self.validator_agent.is_valid(edited):
                        current_molecule = edited
                        break

            # Check if near feasible region
            if self.is_near_feasible(current_molecule, property_constraints):
                candidates.append(current_molecule)

        return candidates

    def select_seed_molecule(self, constraints):
        """Select starting molecule based on constraints"""
        # Use simple building blocks for initial structure
        return "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"  # Ibuprofen as example

    def apply_fragment_edit(self, molecule, position, fragment):
        """Apply fragment replacement at position"""
        # Use RDKit or similar
        mol = Chem.MolFromSmiles(molecule)
        frag = Chem.MolFromSmiles(fragment)

        # Find attachment point
        # Replace fragment at position
        edited = self.perform_substitution(mol, position, frag)

        if edited is None:
            return None

        return Chem.MolToSmiles(edited)

    def is_near_feasible(self, molecule, constraints):
        """Check if molecule is close to satisfying constraints"""
        props = self.compute_properties(molecule)

        distance = 0.0
        for prop_name, (min_val, max_val) in constraints.items():
            current = props[prop_name]
            if current < min_val:
                distance += (min_val - current) ** 2
            elif current > max_val:
                distance += (current - max_val) ** 2

        # Near feasible if within tolerance
        return distance < FEASIBILITY_TOLERANCE
```

### Fragment-Based Reasoning

Use molecular fragments as semantic units for reasoning.

```python
# Fragment-based molecule representation
class FragmentBasis:
    def __init__(self):
        self.fragment_library = {}
        self.fragment_properties = {}

    def decompose_molecule(self, smiles):
        """Break molecule into fragments"""
        mol = Chem.MolFromSmiles(smiles)

        # Use BRICS fragmentation (Breaking of Retrosynthetically Interesting Chemical Substructures)
        frags = BRICS.BRICSDecompose(mol)

        return list(frags)

    def create_property_chains(self, molecule):
        """Create reasoning chains of fragment edits and property deltas"""
        chain = {
            "initial_molecule": molecule,
            "edits": [],
            "property_trajectory": []
        }

        current = molecule
        for step in range(REASONING_DEPTH):
            # Propose edit
            edit = self.propose_edit(current)

            # Record property change
            old_props = self.compute_properties(current)
            new_props = self.compute_properties(edit["result"])
            delta = {k: new_props[k] - old_props[k] for k in old_props}

            chain["edits"].append({
                "operation": edit["operation"],
                "fragment": edit["fragment"],
                "property_delta": delta
            })

            chain["property_trajectory"].append(new_props)

            current = edit["result"]

        return chain

    def compute_properties(self, smiles):
        """Compute physicochemical properties"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        return {
            "QED": Crippen.MolLogP(mol),  # Quantitative Estimate of Druglikeness
            "LogP": Descriptors.MolLogP(mol),
            "MW": Descriptors.MolWt(mol),
            "HOMO": self.estimate_homo(mol),
            "LUMO": self.estimate_lumo(mol)
        }

    def estimate_homo(self, mol):
        """Estimate HOMO using empirical model"""
        # In practice, would use ML model trained on QM9 dataset
        # Simple approximation for now
        n_atoms = mol.GetNumAtoms()
        n_heavy = Descriptors.HeavyAtomCount(mol)
        return -0.3 * n_heavy + 0.5  # Placeholder

    def estimate_lumo(self, mol):
        """Estimate LUMO"""
        # Empirical approximation
        return self.estimate_homo(mol) + 3.5  # Gap approx 3.5 eV for organic molecules
```

### Stage II: Refinement via Group Relative Policy Optimization (GRPO)

Optimize fragment edits using GRPO to minimize property errors.

```python
# Fragment-level optimization with GRPO
class GRPO_Optimizer:
    def __init__(self, policy_model):
        self.policy = policy_model
        self.fragment_basis = FragmentBasis()

    def optimize_fragment_sequence(self, molecule, target_properties, num_edits=5):
        """Use GRPO to optimize fragment editing sequence"""
        current_molecule = molecule
        edit_sequence = []

        for step in range(num_edits):
            # Compute current error
            current_props = self.fragment_basis.compute_properties(current_molecule)
            current_error = self.compute_property_error(current_props, target_properties)

            # Policy: propose fragment edit
            edit_proposal = self.policy.propose_edit(
                current_molecule,
                target_properties,
                current_error
            )

            # Apply edit
            edited_molecule = self.apply_edit(current_molecule, edit_proposal)

            # Compute new error and reward
            new_props = self.fragment_basis.compute_properties(edited_molecule)
            new_error = self.compute_property_error(new_props, target_properties)

            reward = current_error - new_error  # Reward for error reduction

            edit_sequence.append({
                "edit": edit_proposal,
                "molecule": edited_molecule,
                "error_reduction": reward
            })

            current_molecule = edited_molecule

            # Stop if properties satisfied
            if new_error < ERROR_TOLERANCE:
                break

        return {
            "final_molecule": current_molecule,
            "edit_sequence": edit_sequence,
            "final_error": new_error
        }

    def compute_property_error(self, actual_props, target_props):
        """Compute error across all properties"""
        errors = {}
        total_error = 0.0

        for prop_name, (min_val, max_val) in target_props.items():
            current = actual_props[prop_name]

            if current < min_val:
                error = (min_val - current) ** 2
            elif current > max_val:
                error = (current - max_val) ** 2
            else:
                error = 0.0

            errors[prop_name] = error
            total_error += error

        return np.sqrt(total_error / len(target_props))

    def apply_edit(self, molecule, edit_proposal):
        """Apply fragment edit to molecule"""
        # Implementation of fragment substitution
        edited = self.fragment_basis.apply_fragment_edit(
            molecule,
            edit_proposal["position"],
            edit_proposal["fragment"]
        )
        return edited

    def train_policy(self, training_pairs, learning_rate=1e-3):
        """Train policy using GRPO"""
        # Group Relative Policy Optimization
        # Improve policy based on successful edits

        for molecule, target_props in training_pairs:
            # Generate diverse edit proposals
            proposals = self.policy.generate_proposals(molecule, target_props, num=5)

            # Evaluate each proposal
            rewards = []
            for proposal in proposals:
                edited = self.apply_edit(molecule, proposal)
                props = self.fragment_basis.compute_properties(edited)
                error = self.compute_property_error(props, target_props)
                rewards.append(-error)  # Negative error as reward

            # GRPO: rank proposals and update
            ranked_proposals = sorted(zip(proposals, rewards), key=lambda x: x[1], reverse=True)

            for proposal, reward in ranked_proposals:
                # Policy gradient update
                log_prob = self.policy.get_log_prob(proposal)
                loss = -log_prob * reward
                self.policy.backward(loss, learning_rate)
```

### Multi-Property Constraint Satisfaction

Track and report multi-property goal achievement.

```python
# Multi-property validation
class MultiPropertyValidator:
    def __init__(self, property_definitions):
        self.properties = property_definitions

    def validate_molecule(self, smiles, constraints):
        """Check if molecule satisfies all constraints"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"valid": False, "reason": "Invalid SMILES"}

        props = self.compute_properties(mol)

        satisfied = {}
        all_satisfied = True

        for prop_name, (min_val, max_val) in constraints.items():
            value = props[prop_name]
            is_satisfied = min_val <= value <= max_val
            satisfied[prop_name] = {
                "value": value,
                "min": min_val,
                "max": max_val,
                "satisfied": is_satisfied
            }

            if not is_satisfied:
                all_satisfied = False

        return {
            "valid": True,
            "all_constraints_satisfied": all_satisfied,
            "properties": satisfied,
            "compliance_rate": sum(s["satisfied"] for s in satisfied.values()) / len(satisfied)
        }

    def compute_properties(self, mol):
        """Compute all relevant properties"""
        return {
            "QED": Crippen.MolLogP(mol),
            "LogP": Descriptors.MolLogP(mol),
            "MW": Descriptors.MolWt(mol),
            "HOMO": self.estimate_homo(mol),
            "LUMO": self.estimate_lumo(mol),
            "RotBonds": Descriptors.NumRotatableBonds(mol),
            "HBD": Descriptors.NumHDonors(mol),
            "HBA": Descriptors.NumHAcceptors(mol)
        }
```

## Performance Characteristics

- Generates valid molecules with high property constraint satisfaction
- Handles multiple property constraints simultaneously
- Fragment-level reasoning enables interpretability
- GRPO training is computationally efficient

## Integration Pattern

1. Define property constraints (min/max for QED, LogP, MW, HOMO, LUMO)
2. Stage I: Multi-agent fragment editing generates candidates
3. Stage II: GRPO optimizes fragment sequence for property satisfaction
4. Validate final molecule against all constraints
5. Iterate if needed

## Key Insights

- Fragment-level reasoning is more interpretable than atom-level
- Property deltas for fragments enable learning
- Multi-stage approach balances exploration and optimization
- GRPO provides efficient policy learning

## References

- LLMs struggle with numeric property constraints
- Fragment-based representation enables structure-aware reasoning
- Multi-agent collaboration improves exploration
- Group Relative Policy Optimization efficiently trains policies
