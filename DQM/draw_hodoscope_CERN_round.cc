// ─────────────────────────────────────────────────────────────────────────────
//  draw_hodoscope_CERN_round.cc
//
//  Standalone drawer for the TB2026 CERN "round" hodoscope: 2 staggered
//  layers of 16 round (1 mm diameter) fibers per axis, so 32 fibers for X
//  and 32 for Y (64 channels total). Layer 1 covers 0-16 mm, layer 2 is
//  offset by half a fiber and covers 0.5-16.5 mm (no gap between the
//  circles of the two layers) -> active area 16.5x16.5 mm^2, 0.5 mm
//  position resolution.
//
//  Channel naming: RH1X1..RH1X16 / RH1Y1..RH1Y16 (round hodo, layer 1),
//  RH2X1..RH2X16 / RH2Y1..RH2Y16 (layer 2).
//
//  Per event:
//    1. Compute IntADC and PeakADC for all 64 channels (pedestal-corrected).
//    2. Reject fibers whose RAW (pre-normalization) ADC is at/below their
//       configured per-fiber threshold ("over pedestal" hit requirement).
//       Default threshold is 0 for every channel until a calibration run
//       tunes them.
//    3. Apply the per-channel gain-normalization constant to the surviving
//       candidates only.
//    4. Search the max-ADC fiber across BOTH layers together for X (32
//       candidates) and independently for Y (32 candidates) -- i.e. if a
//       layer-1 and a layer-2 fiber both fire, the higher-ADC one wins.
//       If no X survivor or no Y survivor exists, the event is skipped
//       (not filled) for that metric.
//    5. The winning fiber's 1 mm footprint always spans exactly two 0.5 mm
//       bins, so each event fills that 2x2 block of bins (4 Fill calls,
//       weight 1 each). Layer-1 fiber i covers [i-1, i]; layer-2 fiber i
//       covers [i-0.5, i+0.5]. E.g. RH2X3 winning -> X footprint [2.5, 3.5]
//       -> bins [2.5,3) and [3,3.5) filled.
//
//  IntADC and PeakADC decide independently, so the two output histograms
//  can end up with different entry counts.
// ─────────────────────────────────────────────────────────────────────────────

#include "TBread.h"
#include "TButility.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "TH2.h"
#include "TFile.h"

#include "function.h"

namespace fs = std::filesystem;

namespace {

constexpr int kNFibersPerLayer = 16;
constexpr int kNLayers = 2;
constexpr int kNPerAxis = kNLayers * kNFibersPerLayer; // 32
constexpr int kNChannels = 2 * kNPerAxis;               // 64 (X + Y)

// Global channel index layout (0..63):
//   [0..15]  RH1X1..RH1X16
//   [16..31] RH1Y1..RH1Y16
//   [32..47] RH2X1..RH2X16
//   [48..63] RH2Y1..RH2Y16
std::string ChannelName(int idx) {
  const int layer = (idx < kNPerAxis) ? 1 : 2;
  const int within = idx % kNPerAxis;          // 0..31
  const bool isY = (within >= kNFibersPerLayer);
  const int n = (isY ? within - kNFibersPerLayer : within) + 1; // 1..16
  return "RH" + std::to_string(layer) + (isY ? "Y" : "X") + std::to_string(n);
}

// Global index (0..63) for (layer 1|2, axis X|Y, fiber 1..16)
int GlobalIndex(int layer, bool isY, int fiber1based) {
  const int base = (layer == 1) ? 0 : kNPerAxis;
  return base + (isY ? kNFibersPerLayer : 0) + (fiber1based - 1);
}

// Lower edge (mm) of the 1 mm footprint for a fiber. Layer 1 fiber i (1-based)
// covers [i-1, i]; layer 2 fiber i covers [i-0.5, i+0.5].
double FootprintLow(int layer, int fiber1based) {
  const double idx0 = fiber1based - 1;
  return (layer == 1) ? idx0 : (idx0 + 0.5);
}

} // namespace

int main(int argc, char *argv[]) {

  // Usage:
  //   argv[1] = RunNumber            (required)
  //   argv[2] = MaxEvent             (required; -1 = use all)
  //   argv[3] = (optional) normalization constants file. Default:
  //             ./Hodoscope/hodo_norm_CERN_round.txt
  //   argv[4] = (optional) per-fiber threshold file. Default:
  //             ./Hodoscope/hodo_threshold_CERN_round.txt
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <RunNumber> <MaxEvent> [calib.txt] [threshold.txt]\n";
    return 1;
  }

  const int fRunNum = std::stoi(argv[1]);
  int fMaxEvent = std::stoi(argv[2]);

  fs::path dir("./Hodoscope");
  if (!fs::exists(dir)) fs::create_directory(dir);

  TButility util = TButility();
  // Test mapping derived from test_Hodo.csv (round hodo on MID 1=X, 9=Y).
  // NB: path is relative to CWD when running the binary from DQM/.
  util.LoadMapping("./mapping/mapping_test_Hodo.root");

  // Set from the Run 15105 test-mapping average-waveform check; kept in
  // sync with config_general.yml::AUX.Hodoscope.ROUND.RANGE.
  const int first = 200;  // Hodoscope integration range
  const int last  = 600;  // Hodoscope integration range

  // ── Resolve CIDs, skip channels missing from the mapping ────────────────
  std::vector<TBcid> cid(kNChannels);
  std::vector<bool> valid(kNChannels, false);
  for (int i = 0; i < kNChannels; ++i) {
    cid[i] = util.GetCID(ChannelName(i));
    valid[i] = (cid[i].mid() != -1 && cid[i].channel() != -1);
    if (!valid[i]) {
      std::cerr << "[draw_hodoscope_CERN_round] WARNING: channel '"
                << ChannelName(i) << "' not found in mapping; it will read "
                << "as 0 ADC (never wins the max search).\n";
    }
  }

  // ── Per-channel IntADC/PeakADC normalization constants ──────────────────
  // Produced by normalize_hodo_CERN.cc: each line is
  //   "<channel> <entries> <meanInt> <normInt> <meanPeak> <normPeak>"
  // calibrated[ch] = raw[ch] / norm_const[ch]. Missing file/channel -> 1.0.
  // Indexed [layer(0|1)][axis(0=X,1=Y)][fiber 0..15].
  std::array<std::array<std::array<double, kNFibersPerLayer>, 2>, kNLayers> normIntADC;
  std::array<std::array<std::array<double, kNFibersPerLayer>, 2>, kNLayers> normPeakADC;
  for (auto& perLayer : normIntADC)  for (auto& perAxis : perLayer) perAxis.fill(1.0);
  for (auto& perLayer : normPeakADC) for (auto& perAxis : perLayer) perAxis.fill(1.0);

  bool calibLoaded = false;
  {
    const std::string calibPath = (argc >= 4)
        ? std::string(argv[3])
        : std::string("./Hodoscope/hodo_norm_CERN_round.txt");

    std::ifstream in(calibPath);
    if (!in) {
      std::cerr << "[draw_hodoscope_CERN_round] WARNING: calibration file '"
                << calibPath << "' not found.\n"
                << "  -> IntADC/PeakADC will NOT be normalized (all constants = 1).\n"
                << "  -> Run ./normalize_hodo_CERN " << fRunNum
                << "  first, or pass a path as the 3rd argument.\n";
    } else {
      std::string line;
      int nLoaded = 0;
      while (std::getline(in, line)) {
        if (line.empty() || line.front() == '#') continue;
        std::istringstream iss(line);
        std::string ch;
        long entries;
        double meanInt, normInt, meanPeak, normPeak;
        if (!(iss >> ch >> entries >> meanInt >> normInt >> meanPeak >> normPeak)) continue;
        // Channel-name format: "RH1X<n>", "RH1Y<n>", "RH2X<n>", "RH2Y<n>".
        if (ch.size() < 5 || ch[0] != 'R' || ch[1] != 'H') continue;
        const int layer = ch[2] - '0';
        if (layer != 1 && layer != 2) continue;
        const bool isY = (ch[3] == 'Y');
        if (ch[3] != 'X' && ch[3] != 'Y') continue;
        int fiber = 0;
        try { fiber = std::stoi(ch.substr(4)); } catch (...) { continue; }
        if (fiber < 1 || fiber > kNFibersPerLayer) continue;
        const double safeInt  = (normInt  > 1e-9) ? normInt  : 1.0;
        const double safePeak = (normPeak > 1e-9) ? normPeak : 1.0;
        normIntADC[layer - 1][isY ? 1 : 0][fiber - 1] = safeInt;
        normPeakADC[layer - 1][isY ? 1 : 0][fiber - 1] = safePeak;
        ++nLoaded;
      }
      calibLoaded = (nLoaded > 0);
      std::cout << "[draw_hodoscope_CERN_round] Loaded " << nLoaded
                << " normalization constants from " << calibPath << "\n";
    }
  }

  // ── Per-fiber hit thresholds ("over pedestal") ───────────────────────────
  // Compared against the RAW (pre-normalization) ADC. Default 0 for every
  // channel until a calibration run picks proper values per fiber.
  // File format: "<channel> <intADC_thr> <peakADC_thr>"
  std::array<std::array<std::array<double, kNFibersPerLayer>, 2>, kNLayers> thrIntADC;
  std::array<std::array<std::array<double, kNFibersPerLayer>, 2>, kNLayers> thrPeakADC;
  for (auto& perLayer : thrIntADC)  for (auto& perAxis : perLayer) perAxis.fill(0.0);
  for (auto& perLayer : thrPeakADC) for (auto& perAxis : perLayer) perAxis.fill(0.0);
  {
    const std::string thrPath = (argc >= 5)
        ? std::string(argv[4])
        : std::string("./Hodoscope/hodo_threshold_CERN_round.txt");

    std::ifstream in(thrPath);
    if (!in) {
      std::cout << "[draw_hodoscope_CERN_round] NOTICE: threshold file '"
                << thrPath << "' not found; using default threshold = 0 for "
                << "every fiber (no hit rejection).\n";
    } else {
      std::string line;
      int nLoaded = 0;
      while (std::getline(in, line)) {
        if (line.empty() || line.front() == '#') continue;
        std::istringstream iss(line);
        std::string ch;
        double thrInt, thrPeak;
        if (!(iss >> ch >> thrInt >> thrPeak)) continue;
        if (ch.size() < 5 || ch[0] != 'R' || ch[1] != 'H') continue;
        const int layer = ch[2] - '0';
        if (layer != 1 && layer != 2) continue;
        const bool isY = (ch[3] == 'Y');
        if (ch[3] != 'X' && ch[3] != 'Y') continue;
        int fiber = 0;
        try { fiber = std::stoi(ch.substr(4)); } catch (...) { continue; }
        if (fiber < 1 || fiber > kNFibersPerLayer) continue;
        thrIntADC[layer - 1][isY ? 1 : 0][fiber - 1] = thrInt;
        thrPeakADC[layer - 1][isY ? 1 : 0][fiber - 1] = thrPeak;
        ++nLoaded;
      }
      std::cout << "[draw_hodoscope_CERN_round] Loaded " << nLoaded
                << " per-fiber thresholds from " << thrPath << "\n";
    }
  }

  // ── Histograms: 0.5 mm bins over [0, 16.5] mm ────────────────────────────
  constexpr double kActiveSize = 16.5;
  constexpr double kBinWidth = 0.5;
  const int nBins = static_cast<int>(std::round(kActiveSize / kBinWidth)); // 33

  const std::string intADCTitle = std::string("Round Hodoscope IntADC")
      + (calibLoaded ? " (calibrated)" : " (raw, no calib loaded)")
      + ";X[mm];Y[mm];events";
  TH2F* hist_intADC = new TH2F("hodoscope_rnd_intADC", intADCTitle.c_str(),
                                nBins, 0, kActiveSize, nBins, 0, kActiveSize);
  TH2F* hist_peakADC = new TH2F("hodoscope_rnd_peakADC",
                                 "Round Hodoscope PeakADC;X[mm];Y[mm];events",
                                 nBins, 0, kActiveSize, nBins, 0, kActiveSize);

  // Preapare data reader
  // Round hodoscope test setup: MID 1 (X) + MID 9 (Y), Run_15105 test data.
  TBread<TBwaveform> readerWave = TBread<TBwaveform>(
      fRunNum, fMaxEvent, -1, false, "/Volumes/yhep/scratch/YUdaq", {1, 9});

  if (fMaxEvent == -1 || fMaxEvent > readerWave.GetMaxEvent())
    fMaxEvent = readerWave.GetMaxEvent();

  long nFilledIntADC = 0, nFilledPeakADC = 0;

  for (int iEvt = 0; iEvt < fMaxEvent; iEvt++) {
    if (iEvt % 100 == 0) printProgress(iEvt, fMaxEvent);

    TBevt<TBwaveform> anEvt = readerWave.GetAnEvent();

    //////////////////////////////////////////////////////////////////////
    // Raw IntADC / PeakADC for all 64 channels, [layer][axis][fiber]
    //////////////////////////////////////////////////////////////////////
    std::array<std::array<std::array<double, kNFibersPerLayer>, 2>, kNLayers> rawInt{};
    std::array<std::array<std::array<double, kNFibersPerLayer>, 2>, kNLayers> rawPeak{};

    for (int layer = 1; layer <= kNLayers; ++layer) {
      for (int axis = 0; axis < 2; ++axis) { // 0=X, 1=Y
        for (int f = 1; f <= kNFibersPerLayer; ++f) {
          const int gIdx = GlobalIndex(layer, axis == 1, f);
          if (!valid[gIdx]) continue;
          const std::vector<short> wf = anEvt.GetData(cid[gIdx]).waveform();
          rawInt[layer - 1][axis][f - 1]  = GetInt(wf, first, last);
          rawPeak[layer - 1][axis][f - 1] = GetPeak(wf, first, last);
        }
      }
    }

    //////////////////////////////////////////////////////////////////////
    // Above-threshold max search across BOTH layers for a given axis.
    // Returns the winning (layer, fiber) via out-params, or layer=-1 if
    // no candidate passed its threshold.
    //////////////////////////////////////////////////////////////////////
    auto findMax = [&](int axis,
                        const std::array<std::array<std::array<double, kNFibersPerLayer>, 2>, kNLayers>& raw,
                        const std::array<std::array<std::array<double, kNFibersPerLayer>, 2>, kNLayers>& thr,
                        const std::array<std::array<std::array<double, kNFibersPerLayer>, 2>, kNLayers>& norm,
                        int& bestLayer, int& bestFiber) {
      bestLayer = -1;
      bestFiber = -1;
      double bestVal = -1e300;
      for (int layer = 1; layer <= kNLayers; ++layer) {
        for (int f = 1; f <= kNFibersPerLayer; ++f) {
          const double r = raw[layer - 1][axis][f - 1];
          if (r <= thr[layer - 1][axis][f - 1]) continue; // no hit on this fiber
          const double calibrated = r / norm[layer - 1][axis][f - 1];
          if (calibrated > bestVal) { bestVal = calibrated; bestLayer = layer; bestFiber = f; }
        }
      }
    };

    int xLayerInt, xFiberInt, yLayerInt, yFiberInt;
    int xLayerPeak, xFiberPeak, yLayerPeak, yFiberPeak;
    findMax(0, rawInt,  thrIntADC,  normIntADC,  xLayerInt,  xFiberInt);
    findMax(1, rawInt,  thrIntADC,  normIntADC,  yLayerInt,  yFiberInt);
    findMax(0, rawPeak, thrPeakADC, normPeakADC, xLayerPeak, xFiberPeak);
    findMax(1, rawPeak, thrPeakADC, normPeakADC, yLayerPeak, yFiberPeak);

    // IntADC map: only fill if BOTH X and Y had a fiber above threshold.
    if (xLayerInt > 0 && yLayerInt > 0) {
      const double xLow = FootprintLow(xLayerInt, xFiberInt);
      const double yLow = FootprintLow(yLayerInt, yFiberInt);
      hist_intADC->Fill(xLow + 0.25, yLow + 0.25, 1);
      hist_intADC->Fill(xLow + 0.25, yLow + 0.75, 1);
      hist_intADC->Fill(xLow + 0.75, yLow + 0.25, 1);
      hist_intADC->Fill(xLow + 0.75, yLow + 0.75, 1);
      ++nFilledIntADC;
    }
    // PeakADC map: independent decision from IntADC.
    if (xLayerPeak > 0 && yLayerPeak > 0) {
      const double xLow = FootprintLow(xLayerPeak, xFiberPeak);
      const double yLow = FootprintLow(yLayerPeak, yFiberPeak);
      hist_peakADC->Fill(xLow + 0.25, yLow + 0.25, 1);
      hist_peakADC->Fill(xLow + 0.25, yLow + 0.75, 1);
      hist_peakADC->Fill(xLow + 0.75, yLow + 0.25, 1);
      hist_peakADC->Fill(xLow + 0.75, yLow + 0.75, 1);
      ++nFilledPeakADC;
    }
  }
  std::cout << std::endl;
  std::cout << "[draw_hodoscope_CERN_round] Filled IntADC map with "
            << nFilledIntADC << " / " << fMaxEvent << " events, PeakADC map with "
            << nFilledPeakADC << " / " << fMaxEvent << " events (rest skipped: "
            << "no fiber above threshold on X and/or Y).\n";

  //////////////////////////////////////////////////////////////////////
  // Output file
  //////////////////////////////////////////////////////////////////////
  std::string outFile = "./Hodoscope/Hodoscope_CERN_round_Run_" + std::to_string(fRunNum) + ".root";
  TFile* outputRoot = new TFile(outFile.c_str(), "RECREATE");
  outputRoot->cd();

  hist_intADC->Write();
  hist_peakADC->Write();

  outputRoot->Close();
}
