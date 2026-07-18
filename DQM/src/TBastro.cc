#include "TBastro.h"
#include "TBAstroReader.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "TBufferJSON.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TH2.h"

TBastro::TBastro(const YAML::Node& fNodeAstro_, int fRunNum_, const std::string& fGlobalBaseDir_)
    : fRunNum(fRunNum_),
      fMID(1),
      fTsDiff(2),
      fTotDiffPct(10.0),
      fBaseDir(fGlobalBaseDir_),
      fHasData(false),
      fNEvents(0),
      fNHits(0),
      fNPairs(0) {
  for (auto& col : fCounts) col.fill(0);

  if (fNodeAstro_ && fNodeAstro_.IsDefined() && !fNodeAstro_.IsNull()) {
    if (fNodeAstro_["MID"] && fNodeAstro_["MID"].IsScalar())
      fMID = fNodeAstro_["MID"].as<int>();
    if (fNodeAstro_["TS_DIFF"] && fNodeAstro_["TS_DIFF"].IsScalar())
      fTsDiff = fNodeAstro_["TS_DIFF"].as<int>();
    if (fNodeAstro_["TOT_DIFF_PCT"] && fNodeAstro_["TOT_DIFF_PCT"].IsScalar())
      fTotDiffPct = fNodeAstro_["TOT_DIFF_PCT"].as<double>();
    if (fNodeAstro_["BaseDirectory"] && fNodeAstro_["BaseDirectory"].IsScalar()) {
      const std::string dir = fNodeAstro_["BaseDirectory"].as<std::string>();
      if (!dir.empty()) fBaseDir = dir;
    }
  }
}

void TBastro::Run(int maxEvent) {
  std::unique_ptr<TBAstroReader> reader;
  try {
    reader = std::make_unique<TBAstroReader>(fRunNum, fMID, fBaseDir);
  } catch (const std::exception& e) {
    std::cout << "[TBastro] WARNING: could not open AstroPix data for Run " << fRunNum
              << ", MID " << fMID << " under '" << fBaseDir << "': " << e.what()
              << " -- Astro plotting skipped." << std::endl;
    return;
  }

  int nEvent = maxEvent;
  if (nEvent == -1 || nEvent > reader->GetMaxEvent()) nEvent = reader->GetMaxEvent();

  std::cout << "[TBastro] Run " << fRunNum << ", MID " << fMID << " -- " << nEvent << " / "
            << reader->GetMaxEvent() << " events available, matching |dts| < " << fTsDiff
            << " and |dtot|/tot_col < " << fTotDiffPct << "%" << std::endl;

  for (int i = 0; i < nEvent; ++i) {
    if (i > 0 && i % 2000 == 0) {
      const double pct = 100.0 * static_cast<double>(i) / static_cast<double>(nEvent);
      // Plain single-shot line (no \r cursor tricks): the Astro thread's
      // progress must not fight the main loop's own cursor-overwrite
      // progress line when both print to the same terminal/pipe.
      std::cout << "Astro " << i << " / " << nEvent << " events (" << pct << " %)" << std::endl;
      std::fflush(stdout);
    }

    TBAstroEvent evt = reader->GetAnEvent();
    fNHits += static_cast<long>(evt.hits.size());

    std::vector<const TBAstroHit*> colHits, rowHits;
    for (const auto& hit : evt.hits) {
      if (hit.tot == 0) continue;  // matches Multiprocess.py's tot_us == 0 skip
      if (hit.isCol) colHits.push_back(&hit);
      else rowHits.push_back(&hit);
    }

    for (const auto* colHit : colHits) {
      for (const auto* rowHit : rowHits) {
        const int dts = std::abs(static_cast<int>(colHit->ts) - static_cast<int>(rowHit->ts));
        if (dts >= fTsDiff) continue;

        const double dtotPct =
            std::abs(static_cast<double>(colHit->tot) - static_cast<double>(rowHit->tot)) /
            static_cast<double>(colHit->tot) * 100.0;
        if (dtotPct >= fTotDiffPct) continue;

        if (colHit->ch < kNCol && rowHit->ch < kNRow) {
          ++fCounts[colHit->ch][rowHit->ch];
          ++fNPairs;
        }
      }
    }
  }

  fNEvents = nEvent;
  fHasData = true;

  std::cout << "[TBastro] Done: " << fNEvents << " events, " << fNHits << " hits, " << fNPairs
            << " col/row pairs." << std::endl;
}

void TBastro::Update() {
  if (!fHasData) return;

  TH2F* hist = new TH2F("hitmap", "AstroPix Hit Map;Col;Row;Hit pairs", kNCol, 0, kNCol, kNRow, 0,
                         kNRow);
  hist->SetStats(0);
  for (int col = 0; col < kNCol; ++col)
    for (int row = 0; row < kNRow; ++row)
      if (fCounts[col][row] > 0)
        hist->SetBinContent(col + 1, row + 1, static_cast<double>(fCounts[col][row]));

  TCanvas* canvas = new TCanvas("fCanvas_Astro", "AstroPix Hit Map", 700, 600);
  canvas->cd();
  hist->Draw("colz");
  canvas->Update();

  const std::string rootPath = "./output/Run" + std::to_string(fRunNum) + "_Astro.root";
  {
    TFile outFile(rootPath.c_str(), "RECREATE");
    outFile.cd();
    canvas->Write();
    hist->Write();
    outFile.Close();
  }

  // Naming follows TBaux::dumpJSON's convention (Run<N>_<type>_<method>_
  // <canvas>.json) so the web run-browser's existing regex parses it with
  // no server-side changes: type=Astro, method=Hitmap, canvas=fCanvas_Astro.
  const std::string basePrefix = "Run" + std::to_string(fRunNum) + "_Astro_Hitmap";
  const std::string canvasName = canvas->GetName();
  const std::string finalPath = "./output/" + basePrefix + "_" + canvasName + ".json";
  const std::string tmpPath = finalPath + ".tmp";
  {
    std::ofstream ofs(tmpPath);
    if (ofs) {
      TString json = TBufferJSON::ToJSON(canvas);
      ofs << json.Data();
    }
  }
  std::rename(tmpPath.c_str(), finalPath.c_str());

  std::cout << "[TBastro] Wrote " << rootPath << " and " << finalPath << std::endl;
}
