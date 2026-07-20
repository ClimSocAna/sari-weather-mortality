# Third-party data and attribution

The `data/` directory contains analysis-ready derivatives of third-party data.
These files are not relicensed under the repository's MIT or CC BY 4.0 grants.
Users must comply with the applicable provider terms and should consult the
source records for the precise version and any additional notices.

## Mortality and population

Provider: Statistisches Bundesamt (Destatis), with population statistics also
drawn from the statistical offices of the German federal states.

- [Destatis mortality reports](https://www.destatis.de/DE/Themen/Gesellschaft-Umwelt/Bevoelkerung/Sterbefaelle-Lebenserwartung/Publikationen/Downloads-Sterbefaelle/statistischer-bericht-sterbefaelle-tage-wochen-monate-aktuell-5126109.html)
- [GENESIS population table 12411-09-01-4](https://www.regionalstatistik.de/genesis//online?operation=table&code=12411-09-01-4&bypass=true)
- [Data Licence Germany – attribution – version 2.0](https://www.govdata.de/dl-de/by-2-0)

Suggested attribution: Data source: Statistisches Bundesamt (Destatis),
GENESIS-Online and referenced mortality publications; Data Licence Germany –
attribution – version 2.0 (dl-de/by-2-0); modified and processed for this
repository.

## Weather

Provider: Deutscher Wetterdienst (DWD), Climate Data Center, HOSTRADA.

- [DWD HOSTRADA data](https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/)
- [DWD Open Data terms](https://www.dwd.de/DE/leistungen/opendata/faqs_opendata.html)
- [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

The checked-in tensors are temporally aggregated and spatially aggregated or
population-weighted derivatives. Population weighting also uses Destatis 2022
census information and therefore requires the corresponding Destatis source
attribution.

Suggested attribution: Source: Deutscher Wetterdienst (DWD), Climate Data
Center, HOSTRADA; CC BY 4.0; modified and processed for this repository.

## SARI hospitalization incidence

Provider: Robert Koch Institute (RKI).

- [SARI-Hospitalisierungsinzidenz](https://doi.org/10.5281/zenodo.17422294)
- [RKI dataset documentation and license](https://robert-koch-institut.github.io/SARI-Hospitalisierungsinzidenz/)
- [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

Suggested attribution: Robert Koch Institute, SARI-Hospitalisierungsinzidenz;
CC BY 4.0; modified and processed for this repository.

## Administrative boundaries

Source data: Verwaltungsgebiete 1:250 000 (VG250), GeoBasis-DE / Bundesamt
für Kartographie und Geodäsie (BKG), distributed through the Opendatasoft
geographical repository.

- [Germany district boundaries](https://public.opendatasoft.com/explore/dataset/georef-germany-kreis/)
- [Germany state boundaries](https://public.opendatasoft.com/explore/dataset/georef-germany-land/)
- [Data Licence Germany – attribution – version 2.0](https://www.govdata.de/dl-de/by-2-0)

Suggested attribution: GeoBasis-DE / BKG (2020), VG250; Data Licence Germany –
attribution – version 2.0 (dl-de/by-2-0); geometries simplified and processed
by Opendatasoft and further processed for this repository.

## No endorsement

Use of provider names is solely for attribution and does not imply that
Destatis, DWD, RKI, BKG, the statistical offices of the federal states, or
Opendatasoft endorse this repository, manuscript, or its conclusions.
