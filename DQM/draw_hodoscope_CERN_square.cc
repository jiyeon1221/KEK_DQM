// ─────────────────────────────────────────────────────────────────────────────
//  draw_hodoscope_CERN_square.cc
//
//  Standalone drawer for the TB2026 CERN "square" hodoscope: 29 X-fibers +
//  29 Y-fibers (1 mm square cross-section fibers, cross "+" geometry, same
//  concept as the KEK 16x16 hodoscope but larger). Active area 29x29 mm^2,
//  1 mm position resolution.
//
//  Channel naming: SHX1..SHX29 (Square Hodo X 1..29), SHY1..SHY29.
//
//  Per event:
//    1. Compute IntADC and PeakADC for all 58 channels (pedestal-corrected,
//       same GetInt/GetPeak convention as draw_hodoscope.cc).
//    2. Reject fibers whose RAW (pre-normalization) ADC is at/below their
//       configured per-fiber threshold ("over pedestal" hit requirement).
//       Thresholds are loaded from a text file; default is 0 for every
//       channel (no rejection) until a calibration run tunes them.
//    3. Apply the per-channel gain-normalization constant to the surviving
//       candidates only (normalization corrects gain, not efficiency).
//    4. Pick the max-ADC X survivor and max-ADC Y survivor. If either axis
//       has NO survivor, the event is skipped (not filled) for that metric
//       -- this is expected because the hodoscope's per-fiber efficiency is
//       well below 100%, so a fraction of events will have no real hit.
//    5. Fill the 29x29 (1 mm/bin) map at the fiber centers, e.g. SHX3 fired
//       -> x = 2.5 mm (bin [2, 3) center).
//
//  IntADC and PeakADC decide independently (a fiber can pass the IntADC
//  threshold but not the PeakADC one, or vice versa), so the two output
//  histograms can end up with different entry counts.
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

constexpr int kNFibers = 29;   // per axis
constexpr int kNChannels = 2 * kNFibers;

// Returns "SHX<n>" or "SHY<n>" for index 0..57. First 29 are X, next 29 Y.
std::string ChannelName(int idx) {
  const bool isY = (idx >= kNFibers);
  const int n = (isY ? idx - kNFibers : idx) + 1;
  return std::string(isY ? "SHY" : "SHX") + std::to_string(n);
}

} // namespace

int main(int argc, char *argv[]) {

  // Usage:
  //   argv[1] = RunNumber            (required)
  //   argv[2] = MaxEvent             (required; -1 = use all)
  //   argv[3] = (optional) normalization constants file. Produced by
  //             normalize_hodo_CERN.cc. Default:
  //             ./Hodoscope/hodo_norm_CERN_square.txt
  //   argv[4] = (optional) per-fiber threshold file. Default:
  //             ./Hodoscope/hodo_threshold_CERN_square.txt
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <RunNumber> <MaxEvent> [calib.txt] [threshold.txt]\n";
    return 1;
  }

  const int fRunNum = std::stoi(argv[1]);
  int fMaxEvent = std::stoi(argv[2]);

  fs::path dir("./Hodoscope");
  if (!fs::exists(dir)) fs::create_directory(dir);

  // initialize the utility class
  TButility util = TButility();
  // Test mapping derived from test_Hodo.csv (square hodo on MID 13=X, 15=Y).
  // NB: path is relative to CWD when running the binary from DQM/.
  util.LoadMapping("./mapping/mapping_test_Hodo.root");

  // Set from the Run 15105 test-mapping average-waveform check; kept in
  // sync with config_general.yml::AUX.Hodoscope.SQUARE.RANGE.
  const int first = 200;  // Hodoscope integration range
  const int last  = 600;  // Hodoscope integration range

  // ── Resolve CIDs, skip channels missing from the mapping ────────────────
  std::vector<TBcid> cid(kNChannels);
  std::vector<bool> valid(kNChannels, false);
  for (int i = 0; i < kNChannels; ++i) {
    cid[i] = util.GetCID(ChannelName(i));
    valid[i] = (cid[i].mid() != -1 && cid[i].channel() != -1);
    if (!valid[i]) {
      std::cerr << "[draw_hodoscope_CERN_square] WARNING: channel '"
                << ChannelName(i) << "' not found in mapping; it will read "
                << "as 0 ADC (never wins the max search).\n";
    }
  }

  // ── Per-channel IntADC/PeakADC normalization constants ──────────────────
  // Produced by normalize_hodo_CERN.cc: each line is
  //   "<channel> <entries> <meanInt> <normInt> <meanPeak> <normPeak>"
  // calibrated[ch] = raw[ch] / norm_const[ch]. Missing file/channel -> 1.0
  // (no-op), so this binary still runs on uncalibrated data.
  std::array<double, kNFibers> normX_intADC;  normX_intADC.fill(1.0);
  std::array<double, kNFibers> normY_intADC;  normY_intADC.fill(1.0);
  std::array<double, kNFibers> normX_peakADC; normX_peakADC.fill(1.0);
  std::array<double, kNFibers> normY_peakADC; normY_peakADC.fill(1.0);
  bool calibLoaded = false;
  {
    const std::string calibPath = (argc >= 4)
        ? std::string(argv[3])
        : std::string("./Hodoscope/hodo_norm_CERN_square.txt");

    std::ifstream in(calibPath);
    if (!in) {
      std::cerr << "[draw_hodoscope_CERN_square] WARNING: calibration file '"
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
        // Channel-name format: "SHX<n>" or "SHY<n>", n = 1..29.
        if (ch.size() < 4 || ch[0] != 'S' || ch[1] != 'H') continue;
        if (ch[2] != 'X' && ch[2] != 'Y') continue;
        int idx = 0;
        try { idx = std::stoi(ch.substr(3)) - 1; } catch (...) { continue; }
        if (idx < 0 || idx >= kNFibers) continue;
        const double safeInt  = (normInt  > 1e-9) ? normInt  : 1.0;
        const double safePeak = (normPeak > 1e-9) ? normPeak : 1.0;
        if (ch[2] == 'X') { normX_intADC[idx] = safeInt; normX_peakADC[idx] = safePeak; }
        else              { normY_intADC[idx] = safeInt; normY_peakADC[idx] = safePeak; }
        ++nLoaded;
      }
      calibLoaded = (nLoaded > 0);
      std::cout << "[draw_hodoscope_CERN_square] Loaded " << nLoaded
                << " normalization constants from " << calibPath << "\n";
    }
  }

  // ── Per-fiber hit thresholds ("over pedestal") ───────────────────────────
  // Compared against the RAW (pre-normalization) ADC. Default 0 for every
  // channel until a calibration run picks proper values per fiber.
  // File format: "<channel> <intADC_thr> <peakADC_thr>"
  std::array<double, kNFibers> thrX_intADC;  thrX_intADC.fill(0.0);
  std::array<double, kNFibers> thrY_intADC;  thrY_intADC.fill(0.0);
  std::array<double, kNFibers> thrX_peakADC; thrX_peakADC.fill(0.0);
  std::array<double, kNFibers> thrY_peakADC; thrY_peakADC.fill(0.0);
  {
    const std::string thrPath = (argc >= 5)
        ? std::string(argv[4])
        : std::string("./Hodoscope/hodo_threshold_CERN_square.txt");

    std::ifstream in(thrPath);
    if (!in) {
      std::cout << "[draw_hodoscope_CERN_square] NOTICE: threshold file '"
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
        if (ch.size() < 4 || ch[0] != 'S' || ch[1] != 'H') continue;
        if (ch[2] != 'X' && ch[2] != 'Y') continue;
        int idx = 0;
        try { idx = std::stoi(ch.substr(3)) - 1; } catch (...) { continue; }
        if (idx < 0 || idx >= kNFibers) continue;
        if (ch[2] == 'X') { thrX_intADC[idx] = thrInt; thrX_peakADC[idx] = thrPeak; }
        else              { thrY_intADC[idx] = thrInt; thrY_peakADC[idx] = thrPeak; }
        ++nLoaded;
      }
      std::cout << "[draw_hodoscope_CERN_square] Loaded " << nLoaded
                << " per-fiber thresholds from " << thrPath << "\n";
    }
  }

  // ── Histograms ────────────────────────────────────────────────────────────
  const std::string intADCTitle = std::string("Square Hodoscope IntADC")
      + (calibLoaded ? " (calibrated)" : " (raw, no calib loaded)")
      + ";X[mm];Y[mm];events";
  TH2F* hist_intADC = new TH2F("hodoscope_sq_intADC", intADCTitle.c_str(),
                                kNFibers, 0, kNFibers, kNFibers, 0, kNFibers);
  TH2F* hist_peakADC = new TH2F("hodoscope_sq_peakADC",
                                 "Square Hodoscope PeakADC;X[mm];Y[mm];events",
                                 kNFibers, 0, kNFibers, kNFibers, 0, kNFibers);

  // Preapare data reader
  // Square hodoscope test setup: MID 13 (X) + MID 15 (Y), Run_15105 test data.
  TBread<TBwaveform> readerWave = TBread<TBwaveform>(
      fRunNum, fMaxEvent, -1, false, "/Volumes/yhep/scratch/YUdaq", {13, 15});

  if (fMaxEvent == -1 || fMaxEvent > readerWave.GetMaxEvent())
    fMaxEvent = readerWave.GetMaxEvent();

  long nFilledIntADC = 0, nFilledPeakADC = 0;

  for (int iEvt = 0; iEvt < fMaxEvent; iEvt++) {
    if (iEvt % 100 == 0) printProgress(iEvt, fMaxEvent);

    TBevt<TBwaveform> anEvt = readerWave.GetAnEvent();

    //////////////////////////////////////////////////////////////////////
    // Raw IntADC / PeakADC for all 58 channels
    //////////////////////////////////////////////////////////////////////
    std::array<double, kNFibers> rawIntX{}, rawIntY{};
    std::array<double, kNFibers> rawPeakX{}, rawPeakY{};

    for (int i = 0; i < kNFibers; ++i) {
      if (valid[i]) {
        const std::vector<short> wf = anEvt.GetData(cid[i]).waveform();
        rawIntX[i]  = GetInt(wf, first, last);
        rawPeakX[i] = GetPeak(wf, first, last);
      }
      const int jY = kNFibers + i;
      if (valid[jY]) {
        const std::vector<short> wf = anEvt.GetData(cid[jY]).waveform();
        rawIntY[i]  = GetInt(wf, first, last);
        rawPeakY[i] = GetPeak(wf, first, last);
      }
    }

    //////////////////////////////////////////////////////////////////////
    // Above-threshold max search (per metric, per axis). A fiber only
    // enters the max search if its RAW ADC exceeds its own threshold.
    // Normalization is applied only to candidates that already passed.
    //////////////////////////////////////////////////////////////////////
    auto findMax = [](const std::array<double, kNFibers>& raw,
                       const std::array<double, kNFibers>& thr,
                       const std::array<double, kNFibers>& norm,
                       int& bestIdx, double& bestVal) {
      bestIdx = -1;
      bestVal = -1e300;
      for (int i = 0; i < kNFibers; ++i) {
        if (raw[i] <= thr[i]) continue;   // no hit on this fiber
        const double calibrated = raw[i] / norm[i];
        if (calibrated > bestVal) { bestVal = calibrated; bestIdx = i; }
      }
    };

    int xIdxInt, yIdxInt, xIdxPeak, yIdxPeak;
    double xValInt, yValInt, xValPeak, yValPeak;
    findMax(rawIntX,  thrX_intADC,  normX_intADC,  xIdxInt,  xValInt);
    findMax(rawIntY,  thrY_intADC,  normY_intADC,  yIdxInt,  yValInt);
    findMax(rawPeakX, thrX_peakADC, normX_peakADC, xIdxPeak, xValPeak);
    findMax(rawPeakY, thrY_peakADC, normY_peakADC, yIdxPeak, yValPeak);

    // IntADC map: only fill if BOTH X and Y had a fiber above threshold.
    if (xIdxInt >= 0 && yIdxInt >= 0) {
      hist_intADC->Fill(xIdxInt + 0.5, yIdxInt + 0.5, 1);
      ++nFilledIntADC;
    }
    // PeakADC map: independent decision from IntADC.
    if (xIdxPeak >= 0 && yIdxPeak >= 0) {
      hist_peakADC->Fill(xIdxPeak + 0.5, yIdxPeak + 0.5, 1);
      ++nFilledPeakADC;
    }
  }
  std::cout << std::endl;
  std::cout << "[draw_hodoscope_CERN_square] Filled IntADC map with "
            << nFilledIntADC << " / " << fMaxEvent << " events, PeakADC map with "
            << nFilledPeakADC << " / " << fMaxEvent << " events (rest skipped: "
            << "no fiber above threshold on X and/or Y).\n";

  //////////////////////////////////////////////////////////////////////
  // Output file
  //////////////////////////////////////////////////////////////////////
  std::string outFile = "./Hodoscope/Hodoscope_CERN_square_Run_" + std::to_string(fRunNum) + ".root";
  TFile* outputRoot = new TFile(outFile.c_str(), "RECREATE");
  outputRoot->cd();

  hist_intADC->Write();
  hist_peakADC->Write();

  outputRoot->Close();
}
