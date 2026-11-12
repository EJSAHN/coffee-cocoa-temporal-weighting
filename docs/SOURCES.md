# Data sources and references

## Official production statistics

The production and area records come from IBGE's Produção Agrícola Municipal (PAM), accessed through SIDRA. Preserve the original table, variable, crop, municipality, year, unit, response symbols, and extraction metadata when assembling an input bundle.

- [PAM tables and methodological notes](https://sidra.ibge.gov.br/pesquisa/pam/tabelas/)
- [SIDRA table 5457](https://sidra.ibge.gov.br/tabela/5457): temporary and permanent crops.
- [SIDRA table 1613](https://sidra.ibge.gov.br/tabela/1613): permanent crops.
- [IBGE aggregate API documentation](https://servicodados.ibge.gov.br/api/docs/agregados?versao=3)
- [PAM technical notes, 2023 edition](https://www.ibge.gov.br/biblioteca/visualizacao/periodicos/66/pam_2023_v50_br_notas_tecnicas.pdf)

IBGE's notes distinguish intended harvest area from harvested area. They also identify a coffee-product definition change before the 2002 start of this study's production period. A current download may differ from the preserved extraction through source revisions. Differences between source-table symbols are recorded as source discordance rather than silently replacing the original values.

## Monthly climate

- [TerraClimate project and version documentation](https://www.climatologylab.org/terraclimate.html)
- Preserved observed-data OPeNDAP endpoint pattern:
  `https://thredds.northwestknowledge.net/thredds/dodsC/agg_terraclimate_{variable}_1950_CurrentYear_GLOBE.nc`, with variables `ppt`, `pet`, and `tmax`.
- Abatzoglou JT, Dobrowski SZ, Parks SA, Hegewisch KC. 2018. TerraClimate, a high-resolution global dataset of monthly climate and climatic water balance from 1958–2015. *Scientific Data* 5:170191. [doi:10.1038/sdata.2017.191](https://doi.org/10.1038/sdata.2017.191).

The preserved study extraction uses TerraClimate V1.1 observations for January 1980–December 2025. The original publication describes the dataset's foundation; the V1.1 documentation and preserved source metadata are needed to identify the actual version used. The observation product must be distinguished from warming scenarios and the modeled counterfactual product. Preserve packed-data scale factors, temperature offsets, grid coordinates, units, metadata, and file hashes. The analysis command consumes local decoded inputs and does not retrieve these endpoints.

## Statistical references

1. Hsiang S. 2016. Climate econometrics. *Annual Review of Resource Economics* 8:43–75. [Working paper](https://www.nber.org/papers/w22181). Background on climate-panel identification and interpretation.
2. Driscoll JC, Kraay AC. 1998. Consistent covariance matrix estimation with spatially dependent panel data. *Review of Economics and Statistics* 80:549–560. [doi:10.1162/003465398557825](https://doi.org/10.1162/003465398557825). Background for covariance estimation with spatial and temporal dependence; asymptotic results do not ensure accurate coverage with a short annual record.
3. Newey WK, West KD. 1987. A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica* 55:703–708. [doi:10.2307/1913610](https://doi.org/10.2307/1913610). Background for the HAC construction.
4. Morris TP, White IR, Crowther MJ. 2019. Using simulation studies to evaluate statistical methods. *Statistics in Medicine* 38:2074–2102. [doi:10.1002/sim.8086](https://doi.org/10.1002/sim.8086). Framework for distinguishing simulation aims, generating mechanisms, estimands, methods, performance measures, and Monte Carlo error.

These references motivate the design and interpretation. They do not validate the selected empirical specification, source harmonization, or coverage of intervals in the observed study.
