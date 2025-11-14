@echo off
setlocal enabledelayedexpansion

set "BUCKET=gs://plant-ai-dataset-mohamed-20100116"
set "PREFIX=users/Cabinet"

for %%F in (
  Mangrove_agb_Angola.tif
  Mangrove_agb_Benin.tif
  Mangrove_agb_Cameroon.tif
  Mangrove_agb_CoteDivoire.tif
  Mangrove_agb_DemocraticRepublicOfCongo.tif
  Mangrove_agb_Djibouti.tif
  Mangrove_agb_Egypt.tif
  Mangrove_agb_EquatorialGuinea.tif
  Mangrove_agb_Gabon.tif
  Mangrove_agb_Gambia.tif
  Mangrove_agb_Ghana.tif
  Mangrove_agb_Guinea.tif
  Mangrove_agb_Guinea.tif.sha256
  Mangrove_agb_GuineaBissau.tif
  Mangrove_agb_Kenya.tif
  Mangrove_agb_Liberia.tif
  Mangrove_agb_Madagascar.tif
  Mangrove_agb_Mauritania.tif
  Mangrove_agb_Mozambique.tif
  Mangrove_agb_Nigeria.tif
  Mangrove_agb_Nigeria.tif.sha256
  Mangrove_agb_Senegal.tif
  Mangrove_agb_SierraLeone.tif
  Mangrove_agb_Somalia.tif
  Mangrove_agb_Somalia2.tif
  Mangrove_agb_Soudan.tif
  Mangrove_agb_SouthAfrica.tif
  Mangrove_agb_Tanzania.tif
  Mangrove_agb_Togo.tif
  Mangrove_agb_Togo.tif.sha256
  Mangrove_hba95_Angola.tif
  Mangrove_hba95_Angola.tif.sha256
  Mangrove_hba95_Benin.tif
  Mangrove_hba95_Cameroon.tif
  Mangrove_hba95_Cote.tif
  Mangrove_hba95_Cote.tif.sha256
  Mangrove_hba95_CotedIvoire.tif
  Mangrove_hba95_CotedIvoire.tif.sha256
  Mangrove_hba95_DemocraticRepublicOfCongo.tif
  Mangrove_hba95_Djibouti.tif
  Mangrove_hba95_Djibouti.tif.sha256
  Mangrove_hba95_Egypt.tif
  Mangrove_hba95_EquatorialGuinea.tif
  Mangrove_hba95_Gabon.tif
  Mangrove_hba95_Gabon.tif.sha256
  Mangrove_hba95_Gambia.tif
  Mangrove_hba95_Ghana.tif
  Mangrove_hba95_Guinea.tif
  Mangrove_hba95_GuineaBissau.tif
  Mangrove_hba95_Kenya.tif
  Mangrove_hba95_Liberia.tif
  Mangrove_hba95_Madagascar.tif
  Mangrove_hba95_Mauritania.tif
  Mangrove_hba95_Mozambique.tif
  Mangrove_hba95_Nigeria.tif
  Mangrove_hba95_Senegal.tif
  Mangrove_hba95_SierraLeone.tif
  Mangrove_hba95_Somalia.tif
  Mangrove_hba95_SouthAfrica.tif
  Mangrove_hba95_Sudan.tif
  Mangrove_hba95_Tanzania.tif
  Mangrove_hba95_Togo.tif
  Mangrove_hmax95_Angola.tif
  Mangrove_hmax95_Benin.tif
  Mangrove_hmax95_Cameroon.tif
  Mangrove_hmax95_CoteDivoire.tif
  Mangrove_hmax95_DemocraticRepublicOfCongo.tif
  Mangrove_hmax95_Djibouti.tif
  Mangrove_hmax95_Egypt.tif
  Mangrove_hmax95_EquatorialGuinea.tif
  Mangrove_hmax95_EquatorialGuinea.tif.sha256
  Mangrove_hmax95_Gabon.tif
  Mangrove_hmax95_Gambia.tif
  Mangrove_hmax95_Ghana.tif
  Mangrove_hmax95_Guinea.tif
  Mangrove_hmax95_GuineaBissau.tif
  Mangrove_hmax95_Kenya.tif
  Mangrove_hmax95_Liberia.tif
  Mangrove_hmax95_Madagascar.tif
  Mangrove_hmax95_Mauritania.tif
  Mangrove_hmax95_Mozambique.tif
  Mangrove_hmax95_Mozambique.tif.sha256
  Mangrove_hmax95_Nigeria.tif
  Mangrove_hmax95_Senegal.tif
  Mangrove_hmax95_SierraLeone.tif
  Mangrove_hmax95_Somalia.tif
  Mangrove_hmax95_Somalia2.tif
  Mangrove_hmax95_Soudan.tif
  Mangrove_hmax95_SouthAfrica.tif
  Mangrove_hmax95_Tanzania.tif
  Mangrove_hmax95_Togo.tif
) do (
  echo 📤 Importing: %%F
  earthengine upload image --asset_id=%PREFIX%/%%~nF %BUCKET%/%%F
)
