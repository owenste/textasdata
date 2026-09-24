#!/usr/bin/env bash
# Download the raw data used by the pilot study into research/data/raw/.
set -euo pipefail
cd "$(dirname "$0")/../data" && mkdir -p raw && cd raw
curl -sSLO https://data.nber.org/data-appendix/w15319/chat.dta
curl -sSL -o mid5.zip https://correlatesofwar.org/wp-content/uploads/MID-5-Data-and-Supporting-Materials.zip
unzip -o -q mid5.zip 'MIDA 5.0.csv' 'MIDB 5.0.csv'
# V-Dem: the vdemdata R package ships the full dataset as an .RData file
tmp=$(mktemp -d)
GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 https://github.com/vdeminstitute/vdemdata "$tmp/vdemdata"
python3 - "$tmp/vdemdata/data/vdem.RData" <<'PY'
import sys, rdata
df = list(rdata.read_rda(sys.argv[1]).values())[0]
cols = ['country_name','country_text_id','COWcode','year','v2clstown','v2clstown_osp','v2x_polyarchy',
        'v2x_rule','v2x_corr','e_gdppc','e_pop','v2stcritrecadm','v2x_libdem','e_miinteco','e_miinterc','e_civil_war']
df[cols].to_csv('vdem_subset.csv', index=False)
PY
rm -rf "$tmp"
# World Bank WDI: electricity T&D losses, electricity use per capita, carrier departures
for ind in EG.ELC.LOSS.ZS EG.USE.ELEC.KH.PC IS.AIR.DPRT; do
  curl -sSL "https://api.worldbank.org/v2/country/all/indicator/$ind?format=json&per_page=20000" -o "wdi_$ind.json"
done
# Plane Crash Info fatal-accident list, as compiled in JohnyPeters/aviation-accidents-dashboard
curl -sSL -o planecrashinfo.csv \
  https://raw.githubusercontent.com/JohnyPeters/aviation-accidents-dashboard/main/data/crashes_data/plane_crash_data.csv
# WDI Database Archives, July 2018 vintage: country-level T&D losses 1960-2014
curl -sSL "https://api.worldbank.org/v2/sources/57/country/all/series/EG.ELC.LOSS.ZS/version/201807/time/all?format=json&per_page=30000" \
  -o arch_201807.json
# OECD ETCR sector regulation indicators, 1975-2023
curl -sSL -A "Mozilla/5.0" -o etcr.xlsx \
  "https://www.oecd.org/content/dam/oecd/en/topics/policy-sub-issues/product-market-regulation/ETCR%20indicator%20values.xlsx"
