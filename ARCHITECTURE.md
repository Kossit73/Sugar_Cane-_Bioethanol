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
- Farming
- Sourcing
- Processing and production routing
- Commercialization
- Costs
- Working capital
- Financing

## Scenario behavior

| Scenario | Farm share | Purchased share | Farm CAPEX |
|---|---:|---:|---:|
| FARM_ONLY | 100% | 0% | 100% |
| BUY_ONLY | 0% | 100% | 0% |
| HYBRID | User-defined | Residual | Scaled by farm share |

## Processing routes

Sugar cane is converted into bioethanol, sugar, and raw bagasse. Bagasse is then routed to cogeneration, animal-feed conversion, or direct sale. Cogeneration separates internal electricity use from grid exports.

## Component financials and consolidation

Annual income and cash-flow economics are produced for Farming, Bioethanol, Sugar, Electricity Generation, Bagasse, and Animal Feed. Internal cane, bagasse, and electricity transfers give each business a standalone operating view. The consolidated statements eliminate both the internal revenue and matching internal cost, then independently calculate tax losses, debt service, working capital, cash flow, and the balance sheet.

The `Component Reconciliation` and `Checks` outputs demonstrate that component revenue and EBITDA reconcile to the consolidated model after eliminations.
