// ─────────────────────────────────────────────────────────────────────────────
//  normalize_hodo_CERN.cc
//
//  Per-channel IntADC + PeakADC normalization for the two TB2026 CERN
//  hodoscopes:
//    - "square": SHX1..29 / SHY1..29           (58 channels, MID 28)
//    - "round" : RH1X1..16 / RH1Y1..16 /
//                RH2X1..16 / RH2Y1..16          (64 channels, MID 29)
//
//  For every channel:
//    1. Computes IntADC and PeakADC per event over a user-defined
//       integration window (kIntFirst, kIntLast) -- same convention as
//       draw_hodoscope_CERN_square.cc / draw_hodoscope_CERN_round.cc.
//    2. Accumulates per-channel IntADC and PeakADC distributions.
//    3. Takes each histogram's MEAN as that channel's normalization
//       constant (norm_const[ch] = mean[ch]).
//
//  Downstream code applies it as:
//        calibrated[ch] = raw[ch] / norm_const[ch]
//  This is unrelated to the per-fiber HIT THRESHOLD (see
//  hodo_threshold_CERN_square.txt / _round.txt, produced/edited separately):
//  normalization corrects gain, thresholds decide whether a fiber fired at
//  all. Both are applied by the draw_hodoscope_CERN_*.cc binaries and by
//  TBaux in the live DQM.
//
//  Outputs (under ./Hodoscope/):
//    hodo_norm_CERN_square.root / .txt
//    hodo_norm_CERN_round.root  / .txt
//  Each .txt line: "<channel> <entries> <meanInt> <normInt> <meanPeak> <normPeak>"
//  (norm == mean, kept as a separate column so the file is self-describing
//  and matches exactly what draw_hodoscope_CERN_*.cc parses.)
//
//  Usage:
//    ./normalize_hodo_CERN <RunNumber> [MaxEvent=30000]
//        MaxEvent = -1 uses every available event.
// ─────────────────────────────────────────────────────────────────────────────

#include "TBread.h"
#include "TButility.h"
#include "function.h"

#include "TH1.h"
#include "TFile.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kIntFirst = 150;  // TODO: update once real waveform data is available
constexpr int kIntLast  = 350;  // TODO: update once real waveform data is available
constexpr int kDefaultMaxEvent = 30000;

// One entry per channel to process: DAQ name + which text-file "slot" it
// belongs to (both hodos share the same processing loop below).
struct ChannelSpec {
  std::string name;
};

std::vector<ChannelSpec> SquareChannels() {
  std::vector<ChannelSpec> v;
  for (int i = 1; i <= 29; ++i) v.push_back({"SHX" + std::to_string(i)});
  for (int i = 1; i <= 29; ++i) v.push_back({"SHY" + std::to_string(i)});
  return v;
}

std::vector<ChannelSpec> RoundChannels() {
  std::vector<ChannelSpec> v;
  for (int layer = 1; layer <= 2; ++layer) {
    for (int i = 1; i <= 16; ++i) v.push_back({"RH" + std::to_string(layer) + "X" + std::to_string(i)});
    for (int i = 1; i <= 16; ++i) v.push_back({"RH" + std::to_string(layer) + "Y" + std::to_string(i)});
  }
  return v;
}

// Runs the full normalize-and-write pipeline for one hodoscope.
// `mid` is the placeholder DAQ MID that all of this hodoscope's channels
// live on (see mapping_TB2026_v1.csv: 28 = square, 29 = round).
void ProcessHodoscope(const std::string& label, const std::vector<ChannelSpec>& channels,
                       int mid, TButility& util, int fRunNum, int fMaxEvent) {
  const int nCh = static_cast<int>(channels.size());

  std::vector<TBcid> cids(nCh);
  std::vector<bool> valid(nCh, false);
  for (int i = 0; i < nCh; ++i) {
    cids[i] = util.GetCID(channels[i].name);
    valid[i] = (cids[i].mid() != -1 && cids[i].channel() != -1);
    if (!valid[i]) {
      std::cerr << "[normalize_hodo_CERN] WARNING: channel '" << channels[i].name
                << "' not found in mapping; it will be skipped (norm_const = 1.0).\n";
    }
  }

  std::vector<TH1F*> hIntADC(nCh, nullptr), hPeakADC(nCh, nullptr);
  for (int i = 0; i < nCh; ++i) {
    hIntADC[i]  = new TH1F(("h_" + channels[i].name + "_intADC").c_str(),
                            (channels[i].name + " IntADC;IntADC;events").c_str(),
                            440, -30000., 300000.);
    hPeakADC[i] = new TH1F(("h_" + channels[i].name + "_peakADC").c_str(),
                            (channels[i].name + " PeakADC;PeakADC;events").c_str(),
                            440, -30000., 300000.);
  }

  // TODO: Update base directory to the actual TB2026 CERN DAQ data location.
  TBread<TBwaveform> reader = TBread<TBwaveform>(
      fRunNum, fMaxEvent, -1, false, "/PATH/TO/TB2026_CERN_Data", {mid});

  int localMaxEvent = fMaxEvent;
  if (localMaxEvent == -1 || localMaxEvent > reader.GetMaxEvent())
    localMaxEvent = reader.GetMaxEvent();

  std::cout << "[normalize_hodo_CERN] (" << label << ") Run " << fRunNum
            << ", MaxEvent = " << localMaxEvent << ", integration window = ["
            << kIntFirst << ", " << kIntLast << ")" << std::endl;

  for (int iEvt = 0; iEvt < localMaxEvent; ++iEvt) {
    if (iEvt % 1000 == 0) printProgress(iEvt, localMaxEvent);

    TBevt<TBwaveform> anEvt = reader.GetAnEvent();

    for (int i = 0; i < nCh; ++i) {
      if (!valid[i]) continue;
      const auto wave = anEvt.GetData(cids[i]).waveform();
      hIntADC[i]->Fill(GetInt(wave, kIntFirst, kIntLast));
      hPeakADC[i]->Fill(GetPeak(wave, kIntFirst, kIntLast));
    }
  }
  std::cout << std::endl;

  std::vector<double> meanInt(nCh, 0.0), meanPeak(nCh, 0.0);
  std::vector<long> N(nCh, 0);
  int nGood = 0;
  for (int i = 0; i < nCh; ++i) {
    if (!valid[i] || hIntADC[i]->GetEntries() == 0) continue;
    meanInt[i]  = hIntADC[i]->GetMean();
    meanPeak[i] = hPeakADC[i]->GetMean();
    N[i] = static_cast<long>(hIntADC[i]->GetEntries());
    ++nGood;
  }

  if (nGood == 0) {
    std::cerr << "[normalize_hodo_CERN] WARNING: (" << label
              << ") no valid channels with entries; writing all-1.0 constants.\n";
  }

  // norm_const[ch] = mean[ch]; channels with no data default to 1.0 so a
  // downstream blind divide is a no-op instead of inf/NaN.
  std::vector<double> normInt(nCh, 1.0), normPeak(nCh, 1.0);
  for (int i = 0; i < nCh; ++i) {
    if (valid[i] && meanInt[i]  != 0.0) normInt[i]  = meanInt[i];
    if (valid[i] && meanPeak[i] != 0.0) normPeak[i] = meanPeak[i];
  }

  // ── Write QA ROOT file ──────────────────────────────────────────────────
  const std::string rootPath = "./Hodoscope/hodo_norm_CERN_" + label + ".root";
  TFile* outRoot = new TFile(rootPath.c_str(), "RECREATE");
  outRoot->cd();
  for (int i = 0; i < nCh; ++i) {
    if (hIntADC[i])  hIntADC[i]->Write();
    if (hPeakADC[i]) hPeakADC[i]->Write();
  }
  outRoot->Close();
  delete outRoot;

  // ── Write plain-text table of constants ────────────────────────────────
  const std::string txtPath = "./Hodoscope/hodo_norm_CERN_" + label + ".txt";
  std::ofstream out(txtPath);
  if (!out) {
    std::cerr << "[normalize_hodo_CERN] ERROR: cannot open " << txtPath << " for writing.\n";
    return;
  }

  out << "# " << label << " hodoscope IntADC/PeakADC normalization constants -- Run " << fRunNum << "\n";
  out << "# Generated from up to " << localMaxEvent << " events; integration window ["
      << kIntFirst << ", " << kIntLast << ").\n";
  out << "# Good (with-data) channels: " << nGood << " / " << nCh << "\n";
  out << "# norm[ch] = mean of channel ch's distribution.\n";
  out << "# Usage: calibrated[ch] = raw[ch] / norm[ch]\n";
  out << "#\n";
  out << "# " << std::left << std::setw(8) << "ch"
              << std::right << std::setw(10) << "entries"
              << std::right << std::setw(14) << "meanInt"
              << std::right << std::setw(14) << "normInt"
              << std::right << std::setw(14) << "meanPeak"
              << std::right << std::setw(14) << "normPeak"
              << "\n";
  out << std::fixed;
  for (int i = 0; i < nCh; ++i) {
    out << "  " << std::left << std::setw(8) << channels[i].name
                << std::right << std::setw(10) << N[i]
                << std::right << std::setw(14) << std::setprecision(2) << meanInt[i]
                << std::right << std::setw(14) << std::setprecision(6) << normInt[i]
                << std::right << std::setw(14) << std::setprecision(2) << meanPeak[i]
                << std::right << std::setw(14) << std::setprecision(6) << normPeak[i]
                << "\n";
  }
  out.close();

  std::cout << "[normalize_hodo_CERN] (" << label << ") wrote " << txtPath
            << " and " << rootPath << " (" << nGood << "/" << nCh << " channels with data)\n";
}

} // namespace

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <RunNumber> [MaxEvent=" << kDefaultMaxEvent << "]\n"
              << "       MaxEvent = -1 uses every available event.\n";
    return 1;
  }

  const int fRunNum = std::stoi(argv[1]);
  const int fMaxEvent = (argc >= 3) ? std::stoi(argv[2]) : kDefaultMaxEvent;

  fs::path outDir("./Hodoscope");
  if (!fs::exists(outDir)) fs::create_directory(outDir);

  TButility util;
  // TODO: Update mapping path if a newer TB2026 mapping revision is produced.
  util.LoadMapping("../mapping/mapping_TB2026_v1.root");

  // MID 28 = square hodoscope (SHX/SHY), MID 29 = round hodoscope (RH1/RH2 X/Y)
  // -- placeholders, see mapping_TB2026_v1.csv.
  ProcessHodoscope("square", SquareChannels(), 28, util, fRunNum, fMaxEvent);
  ProcessHodoscope("round",  RoundChannels(),  29, util, fRunNum, fMaxEvent);

  return 0;
}
