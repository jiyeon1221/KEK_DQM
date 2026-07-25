// ─────────────────────────────────────────────────────────────────────────────
//  draw_astropix_CERN.cc
//
//  Standalone drawer for the AstroPix position detector: builds the 35x35
//  unmasked 2D hit map from a Run's AstroPix .dat data using TBAstroReader
//  (an independent reader — AstroPix's ASIC-defined data layout does not fit
//  the standard TBread waveform/fastmode format).
//
//  Per event:
//    1. Read all decoded hits (already rsv/ch-filtered by TBAstroReader).
//    2. Split into column hits (isCol == 1) and row hits (isCol == 0);
//       drop hits with tot == 0 (matches the reference Multiprocess.py).
//    3. For every (col hit, row hit) pair, accept it if BOTH hold:
//         |ts_col - ts_row| < tsDiff
//         |tot_col - tot_row| / tot_col * 100 < totDiffPct
//       (aid/chip id is ignored when pairing, per the reference logic.)
//    4. Fill the 35x35 hitmap at (ch_col, ch_row) for every accepted pair.
//
//  This is intentionally minimal (unmasked 2D distribution only): ToT
//  distributions, masked-pixel overlays, per-aid maps and TCB trigger-time
//  matching with waveform MIDs are out of scope here (later task, once this
//  is folded into the DQM AUX system).
// ─────────────────────────────────────────────────────────────────────────────

#include "TBAstroReader.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "TFile.h"
#include "TH2.h"

namespace fs = std::filesystem;

namespace {
constexpr int kNCol = 35;
constexpr int kNRow = 35;
}  // namespace

int main(int argc, char* argv[]) {
  // Usage:
  //   argv[1] = RunNumber            (required)
  //   argv[2] = MaxEvent              (required; -1 = use all)
  //   argv[3] = (optional) tsDiff, default 2      (|ts_col - ts_row| < tsDiff)
  //   argv[4] = (optional) totDiffPct, default 10 (% ToT difference limit)
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <RunNumber> <MaxEvent> [tsDiff=2] [totDiffPct=10]\n";
    return 1;
  }

  const int fRunNum = std::stoi(argv[1]);
  int fMaxEvent = std::stoi(argv[2]);
  const int tsDiff = (argc >= 4) ? std::stoi(argv[3]) : 2;
  const double totDiffPct = (argc >= 5) ? std::stod(argv[4]) : 10.0;

  fs::path dir("./AstroPix");
  if (!fs::exists(dir)) fs::create_directory(dir);

  // TODO: MID and base directory are placeholders for the toy data set.
  // Update the base directory to the real TB2026 CERN DAQ data path once
  // available (same convention as the other draw_*_CERN.cc scripts).
  const int fMID = 1;
  const std::string baseDir = "../toy_astroPix";

  TBAstroReader reader(fRunNum, fMID, baseDir);

  if (fMaxEvent == -1 || fMaxEvent > reader.GetMaxEvent()) fMaxEvent = reader.GetMaxEvent();

  std::cout << "[draw_astropix_CERN] Run " << fRunNum << ", MID " << fMID << " — "
            << fMaxEvent << " / " << reader.GetMaxEvent() << " events, matching within |dts| < "
            << tsDiff << " and |dtot|/tot_col < " << totDiffPct << "%\n";

  TH2F* hist_hitmap = new TH2F("hitmap", "AstroPix Hit Map;Col;Row;Hit pairs", kNCol, 0, kNCol,
                                kNRow, 0, kNRow);
  hist_hitmap->SetStats(0);

  long nHits = 0;
  long nPairs = 0;

  for (int iEvt = 0; iEvt < fMaxEvent; ++iEvt) {
    if (iEvt % 5000 == 0)
      std::cout << "  event " << iEvt << " / " << fMaxEvent << "\r" << std::flush;

    TBAstroEvent evt = reader.GetAnEvent();
    nHits += static_cast<long>(evt.hits.size());

    std::vector<const TBAstroHit*> colHits, rowHits;
    for (const auto& hit : evt.hits) {
      if (hit.tot == 0) continue;  // matches Multiprocess.py's tot_us == 0 skip
      if (hit.isCol) colHits.push_back(&hit);
      else rowHits.push_back(&hit);
    }

    for (const auto* colHit : colHits) {
      for (const auto* rowHit : rowHits) {
        const int dts = std::abs(static_cast<int>(colHit->ts) - static_cast<int>(rowHit->ts));
        if (dts >= tsDiff) continue;

        const double dtotPct =
            std::abs(static_cast<double>(colHit->tot) - static_cast<double>(rowHit->tot)) /
            static_cast<double>(colHit->tot) * 100.0;
        if (dtotPct >= totDiffPct) continue;

        hist_hitmap->Fill(colHit->ch, rowHit->ch);
        ++nPairs;
      }
    }
  }
  std::cout << std::endl;

  std::cout << "[draw_astropix_CERN] Read " << fMaxEvent << " events, decoded " << nHits
            << " hits, filled " << nPairs << " col/row pairs into the hit map.\n";

  const std::string outFile = "./AstroPix/AstroPix_Run_" + std::to_string(fRunNum) + ".root";
  TFile* outputRoot = new TFile(outFile.c_str(), "RECREATE");
  outputRoot->cd();

  hist_hitmap->Write();

  outputRoot->Close();

  std::cout << "[draw_astropix_CERN] Wrote " << outFile << std::endl;

  return 0;
}
