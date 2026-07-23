# Sugar Cane Bioethanol Architecture

The refactored model follows the Cassava Ethanol separation of concerns:

1. `sugarcane_model/inputs.py` defines the grouped input landing page.
2. `sugarcane_model/schedules.py` builds operating and finance schedules.
3. `sugarcane_model/financial_model.py` validates, sequences, caches, and assembles a scenario.
4. `sugarcane_model/scenario.py` compares FARM_ONLY, BUY_ONLY, and HYBRID.
5. `sugarcane_model/exporter.py` produces the full Excel pack and PDF summary.
6. `sugarcane_refactored_plugin.py` provides the Streamlit UI adapter used by NumQuants.
7. `cassava_streamlit_app.py` supplies report downloads and saved-case state while the legacy entry point delegates to the modular application.

## Input sections

- Global assumptions
- Other assumptions
- Capex
- Cycle planning
- Farm planning
- Farming
- Sourcing
- Processing and production routing
- Commercialization
- Costs
- Working capital
- Financing

## Scenario behavior

| Scenario | Own-farm cane | Purchased cane | Farm-share input | Farm CAPEX |
|---|---:|---:|---:|---:|
| FARM_ONLY | Land-plan constrained | 0% | 100% target | 100% |
| BUY_ONLY | 0% | Full requirement | 0% target | 0% |
| HYBRID | Land-plan constrained | Residual requirement | Target/check | Scaled by actual farm share |

## Farm planning

The `Farm Planning` driver schedule has one effective row per projection year. It owns the physical land envelope and annual hectare deployment: total land, arable/cultivable land, planned cultivated area, irrigation capacity, planned irrigated area, and hectares harvested. Rain-fed hectares and fallow/reserve land are calculated from the land envelope. Hectares replanted are calculated from the harvested area and the separate Cycle Planning replant-share assumptions.

Own-farm cane availability is calculated as:

`hectares harvested x cane yield x harvest recovery`

Farm cane processed cannot exceed either harvested cane availability or the plant's cane requirement. HYBRID and BUY_ONLY automatically purchase the residual requirement; FARM_ONLY records any feedstock shortfall as processing underutilization. The model checks all land-capacity and feedstock reconciliations in both the input validator and the model-check output.


## Processing routes

Sugar cane is converted into bioethanol, sugar, and raw bagasse. Bagasse is then routed to cogeneration, animal-feed conversion, or direct sale. Cogeneration separates internal electricity use from grid exports.

## Component financials and consolidation

Annual income and cash-flow economics are produced for Farming, Bioethanol, Sugar, Electricity Generation, Bagasse, and Animal Feed. Internal cane, bagasse, and electricity transfers give each business a standalone operating view. The consolidated statements eliminate both the internal revenue and matching internal cost, then independently calculate tax losses, debt service, working capital, cash flow, and the balance sheet.

The `Component Reconciliation` and `Checks` outputs demonstrate that component revenue and EBITDA reconcile to the consolidated model after eliminations.
