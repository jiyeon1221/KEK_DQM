#include "TBaux.h"
#include "GuiTypes.h"
#include "TSystem.h"
#include "TStyle.h"
#include "TBufferJSON.h"
#include <sys/types.h>
#include <fstream>
#include <cstdio>


TBaux::TBaux(const YAML::Node fNodePlot_, int fRunNum_, bool fPlotting_, bool fLive_, bool fDraw_, TButility fUtility_)
: fNodeAux(fNodePlot_),
  fRunNum(fRunNum_),
  fPlotting(fPlotting_),
  fLive(fLive_),
  fDraw(fDraw_),
  fAuxCut(false),
  fAuxCutMode("WC"),
  fInclinationCut({4.0, 4.0}),
  fUtility(fUtility_),
  fApp(nullptr),
  fCanvas(nullptr),
  fIsFirst(true),
  fMethod(""),
  fWCPosition(nullptr),
  fWCThreshold(0.3),
  fWCCalibration(0.05),
  fWCReference(2, 0.),
  fWCPosCut(-1.),
  fWCEnabled(false),
  fCID_WCX(),
  fCID_WCY(),
  fCID_NIM(),
  fHodoEnabled(false),
  fCID_HodoX(),
  fCID_HodoY(),
  fHodoFirstX(16, 150),
  fHodoLastX(16, 350),
  fHodoFirstY(16, 150),
  fHodoLastY(16, 350),
  fHodoCenter(2, 8.0f),
  fHodoCutMethod("IntADC"),
  fHodoIntADC(nullptr),
  fHodoPeakADC(nullptr),
  fCanvasHodoIntADC(nullptr),
  fCanvasHodoPeakADC(nullptr),
  fHodoIntADC_corr(nullptr),
  fHodoPeakADC_corr(nullptr),
  fCanvasHodoIntADC_corr(nullptr),
  fCanvasHodoPeakADC_corr(nullptr)
{}

void TBaux::init() {

  const auto nodeWC = fNodeAux["WC"];
  if (!nodeWC) {
    throw std::runtime_error("AUX.WC is not configured in YAML.");
  }

  if (nodeWC["CALIB"]) {
    fWCCalibration = nodeWC["CALIB"].as<double>();
  } else {
    throw std::runtime_error("AUX.WC.CALIB is missing in YAML.");
  }

  if (nodeWC["CENTER"]) {
    fWCReference = nodeWC["CENTER"].as<std::vector<double>>();
    if (fWCReference.size() != 2) {
      throw std::runtime_error("AUX.WC.CENTER must contain exactly two values (X_ref, Y_ref).");
    }
  } else {
    throw std::runtime_error("AUX.WC.CENTER is missing in YAML.");
  }

  if (nodeWC["THRESHOLD"])
    fWCThreshold = nodeWC["THRESHOLD"].as<double>();

  if (nodeWC["POSCUT"])
    fWCPosCut = nodeWC["POSCUT"].as<double>();

  fCIDtoPlot.clear();
  fCID_WCX = fUtility.GetCID("WCX");
  fCID_WCY = fUtility.GetCID("WCY");
  fCID_NIM = fUtility.GetCID("NIM");

  // TButility::GetCID() returns TBcid(-1, -1) when the channel name is not
  // present in the loaded mapping. Defer enabling WC processing until we
  // know every WC channel resolved — otherwise Fill() and IsPassing() will
  // call anEvent.GetData(TBcid(-1, -1)).waveform() and throw out_of_range.
  auto cidValid = [](const TBcid& c) { return c.mid() >= 0 && c.channel() >= 0; };
  fWCEnabled = cidValid(fCID_WCX) && cidValid(fCID_WCY) && cidValid(fCID_NIM);
  if (!fWCEnabled) {
    std::cout << "[TBaux] WC channels missing from the loaded mapping "
              << "(WCX/WCY/NIM); --AUX plots and --AUXcut will be skipped."
              << std::endl;
  }

  fCIDtoPlot.push_back(fCID_WCX);
  fCIDtoPlot.push_back(fCID_WCY);
  fCIDtoPlot.push_back(fCID_NIM);

  fWCPosition = new TH2D(
    "WC_Position",
    (TString)"Run " + std::to_string(fRunNum) + " Wire Chamber position;X [mm];Y [mm]",
    120, -30., 30., 120, -30., 30.);
  fWCPosition->SetStats(0);

  fCanvas = new TCanvas("fCanvas_WC", "fCanvas_WC", 1200, 800);
  fCanvas->Divide(1, 1);
  fCanvas->cd(1)->SetRightMargin(0.13);

  // ── Hodoscope ─────────────────────────────────────────────────────────
  // 16 X-fibers and 16 Y-fibers. Channel names must exist in the mapping
  // file pointed to by config_general.yml::Mapping (e.g. toymapping_v2.root
  // contains HX1..HX16 / HY1..HY16). The order here defines the fiber
  // index 0..15 used in the 2D hit-map plots — keep it strictly sequential
  // so bin N == fiber HX(N+1).
  const std::vector<std::string> hodoX_names = {
    "HX1","HX2","HX3","HX4","HX5","HX6","HX7","HX8",
    "HX9","HX10","HX11","HX12","HX13","HX14","HX15","HX16"
  };
  const std::vector<std::string> hodoY_names = {
    "HY1","HY2","HY3","HY4","HY5","HY6","HY7","HY8",
    "HY9","HY10","HY11","HY12","HY13","HY14","HY15","HY16"
  };

  fCID_HodoX.clear();
  fCID_HodoY.clear();
  fCID_HodoX.reserve(hodoX_names.size());
  fCID_HodoY.reserve(hodoY_names.size());
  for (const auto& n : hodoX_names) fCID_HodoX.push_back(fUtility.GetCID(n));
  for (const auto& n : hodoY_names) fCID_HodoY.push_back(fUtility.GetCID(n));

  // Only enable the hodoscope path if every HX/HY name resolved. Mixing
  // valid and invalid CIDs would crash GetHodoscopeRawPosition() the same
  // way the WC code would crash with missing CIDs.
  bool hodoAllValid = true;
  for (const auto& cid : fCID_HodoX) hodoAllValid &= cidValid(cid);
  for (const auto& cid : fCID_HodoY) hodoAllValid &= cidValid(cid);
  if (!hodoAllValid) {
    std::cout << "[TBaux] Hodoscope channels missing from the loaded mapping "
              << "(HX1..HX16 / HY1..HY16); hodoscope AUX plots will be skipped."
              << std::endl;
  }

  // Make sure TBread fetches the MIDs hosting these channels too.
  for (const auto& cid : fCID_HodoX) fCIDtoPlot.push_back(cid);
  for (const auto& cid : fCID_HodoY) fCIDtoPlot.push_back(cid);

  fHodoIntADC = new TH2F(
    "hodoscope_intADC",
    (TString)"Run " + std::to_string(fRunNum) + " Hodoscope IntADC;X [fiber];Y [fiber];events",
    16, 0., 16., 16, 0., 16.);
  fHodoIntADC->SetStats(0);

  fHodoPeakADC = new TH2F(
    "hodoscope_peakADC",
    (TString)"Run " + std::to_string(fRunNum) + " Hodoscope PeakADC;X [fiber];Y [fiber];events",
    16, 0., 16., 16, 0., 16.);
  fHodoPeakADC->SetStats(0);

  fCanvasHodoIntADC = new TCanvas("fCanvas_HodoIntADC", "fCanvas_HodoIntADC", 800, 800);
  fCanvasHodoIntADC->cd()->SetRightMargin(0.13);

  fCanvasHodoPeakADC = new TCanvas("fCanvas_HodoPeakADC", "fCanvas_HodoPeakADC", 800, 800);
  fCanvasHodoPeakADC->cd()->SetRightMargin(0.13);

  // ── Center-corrected hodoscope hit maps ───────────────────────────────
  // Read AUX.Hodoscope.CENTER (the raw fiber-coordinate position where the
  // beam-center actually falls). The corrected plot shifts the beam onto
  // the nominal hodoscope center (8, 8): corrected = raw - CENTER + (8, 8).
  // When CENTER is left at its default of [8, 8] the correction is a no-op
  // and the corrected histograms are identical to the raw ones.
  const auto nodeHodo = fNodeAux["Hodoscope"];
  if (nodeHodo && nodeHodo["CENTER"]) {
    const auto c = nodeHodo["CENTER"].as<std::vector<float>>();
    if (c.size() == 2) fHodoCenter = c;
  }
  if (nodeHodo && nodeHodo["CUT_METHOD"]) {
    const auto m = nodeHodo["CUT_METHOD"].as<std::string>();
    if (m == "IntADC" || m == "PeakADC") {
      fHodoCutMethod = m;
    } else {
      std::cout << "[TBaux] Unrecognised AUX.Hodoscope.CUT_METHOD '" << m
                << "'. Falling back to '" << fHodoCutMethod << "'." << std::endl;
    }
  }

  // Inclination cut [X, Y] in mm, applied only in --AUXCutMode WCHodo.
  // Kept at AUX-top-level because the cut spans both subsystems.
  if (fNodeAux["INCLINATION_CUT"]) {
    const auto v = fNodeAux["INCLINATION_CUT"].as<std::vector<double>>();
    if (v.size() == 2) fInclinationCut = v;
  }

  fHodoIntADC_corr = new TH2F(
    "hodoscope_intADC_corr",
    (TString)"Run " + std::to_string(fRunNum) + " Hodoscope IntADC (center-corrected);X [fiber];Y [fiber];events",
    16, 0., 16., 16, 0., 16.);
  fHodoIntADC_corr->SetStats(0);

  fHodoPeakADC_corr = new TH2F(
    "hodoscope_peakADC_corr",
    (TString)"Run " + std::to_string(fRunNum) + " Hodoscope PeakADC (center-corrected);X [fiber];Y [fiber];events",
    16, 0., 16., 16, 0., 16.);
  fHodoPeakADC_corr->SetStats(0);

  fCanvasHodoIntADC_corr = new TCanvas("fCanvas_HodoIntADC_corr", "fCanvas_HodoIntADC_corr", 800, 800);
  fCanvasHodoIntADC_corr->cd()->SetRightMargin(0.13);

  fCanvasHodoPeakADC_corr = new TCanvas("fCanvas_HodoPeakADC_corr", "fCanvas_HodoPeakADC_corr", 800, 800);
  fCanvasHodoPeakADC_corr->cd()->SetRightMargin(0.13);

  fHodoEnabled = hodoAllValid;
}

void TBaux::SetParticle(std::string fParticle_) {

  fParticle = fParticle_;

  // if (fParticle == "PION") {
  //   fCC1cut = fNodeAux["PION"]["CC1"].as<double>(); 
  //   fCC2cut = fNodeAux["PION"]["CC2"].as<double>();
  //   fPSInitCut = fNodeAux["PION"]["PS_INIT"].as<double>();
  //   fPSFinCut = fNodeAux["PION"]["PS_FIN"].as<double>();
  // }

  // if (fParticle == "KAON") {
  //   fCC1cut = fNodeAux["KAON"]["CC1"].as<double>();
  //   fCC2cut = fNodeAux["KAON"]["CC2"].as<double>();
  //   fPSInitCut = fNodeAux["KAON"]["PS_INIT"].as<double>();
  //   fPSFinCut = fNodeAux["KAON"]["PS_FIN"].as<double>();
  // }

  // if (fParticle == "PROTON") {
  //   fCC1cut = fNodeAux["PROTON"]["CC1"].as<double>();
  //   fCC2cut = fNodeAux["PROTON"]["CC2"].as<double>();
  //   fPSInitCut = fNodeAux["PROTON"]["PS_INIT"].as<double>();
  //   fPSFinCut = fNodeAux["PROTON"]["PS_FIN"].as<double>();
  // }
}

void TBaux::SetRange(const YAML::Node tConfigNode) {

  // Reserved for future per-channel range configuration (e.g., WCX/WCY windows)
  // Example usage:
  // fRangeMap.insert(std::make_pair("WCX", tConfigNode["WCX"].as<std::vector<int>>()));

  // Per-fiber search windows. Read ModuleConfig.HX1..HX16 / HY1..HY16 (the
  // same entries TBplotengine uses for `--type single --module HX1`, so
  // there is a single source of truth for hodoscope ranges). Each entry is
  // [first, last] and is applied to BOTH the IntADC and PeakADC brightest-
  // fiber scans on that fiber. If a key is missing, the per-fiber default
  // (150, 350) seeded in the constructor is kept.
  for (int i = 0; i < 16; ++i) {
    const std::string nameX = "HX" + std::to_string(i + 1);
    const std::string nameY = "HY" + std::to_string(i + 1);
    if (tConfigNode[nameX]) {
      const auto r = tConfigNode[nameX].as<std::vector<int>>();
      if (r.size() == 2) { fHodoFirstX[i] = r[0]; fHodoLastX[i] = r[1]; }
    }
    if (tConfigNode[nameY]) {
      const auto r = tConfigNode[nameY].as<std::vector<int>>();
      if (r.size() == 2) { fHodoFirstY[i] = r[0]; fHodoLastY[i] = r[1]; }
    }
  }
}

double TBaux::GetPeakADC(std::vector<short> waveform, int xInit, int xFin) {
  double ped = 0;
  for (int i = 1; i < 101; i++)
    ped += (double)waveform.at(i) / 100.;

  std::vector<double> pedCorWave;
  for (int i = xInit; i < xFin; i++)
    pedCorWave.push_back(ped - (double)waveform.at(i));

  return *std::max_element(pedCorWave.begin(), pedCorWave.end());
}

double TBaux::GetIntADC(std::vector<short> waveform, int xInit, int xFin) {
  double ped = 0;
  for (int i = 1; i < 101; i++)
    ped += (double)waveform.at(i) / 100.;

  double intADC_ = 0;
  for (int i = xInit; i < xFin; i++)
    intADC_ += ped - (double)waveform.at(i);

  return intADC_;
}

float TBaux::LinearInterp(float x1, float y1, float x2, float y2, float threshold) const {
  return x1 + (threshold - y1) * (x2 - x1) / (y2 - y1);
}

float TBaux::GetLeadingEdgeBin(const std::vector<float>& waveform, float percent) const {

  if (waveform.size() < 1002)
    return -1;

  float max = *std::max_element(waveform.begin() + 1, waveform.begin() + 1001);
  float thr = max * percent;

  for (int i = 1; i < 1000; i++) {
    if (waveform.at(i) < thr && waveform.at(i + 1) > thr) {
      return LinearInterp(static_cast<float>(i), waveform.at(i), static_cast<float>(i + 1), waveform.at(i + 1), thr);
    }
  }
  return -1; // Return -1 if no crossing is found
}

std::vector<float> TBaux::GetPosition(const std::vector<std::vector<float>>& wave) {

  if (wave.size() < 3)
    return {};

  auto binToTime = [](float bin) {
    return 800.f * (bin / 1000.f);
  };

  const float wcxBin = GetLeadingEdgeBin(wave.at(0), static_cast<float>(fWCThreshold));
  const float wcyBin = GetLeadingEdgeBin(wave.at(1), static_cast<float>(fWCThreshold));
  const float nimBin = GetLeadingEdgeBin(wave.at(2), static_cast<float>(fWCThreshold));

  if (wcxBin < 0 || wcyBin < 0 || nimBin < 0)
    return {};

  const float wcxTime = binToTime(wcxBin);
  const float wcyTime = binToTime(wcyBin);
  const float nimTime = binToTime(nimBin);

  const float timeDiffX = nimTime - wcxTime;
  const float timeDiffY = nimTime - wcyTime;

  const float posX = static_cast<float>((fWCReference.at(0) - timeDiffX) * fWCCalibration);
  const float posY = static_cast<float>(-1. * (fWCReference.at(1) - timeDiffY) * fWCCalibration);

  return {posX, posY};
}

void TBaux::Fill(TBevt<TBwaveform> anEvent) {

  // ── Wire chamber position ───────────────────────────────────────────────
  // Skip when the loaded mapping has no WC channels (fWCEnabled == false);
  // GetData on an invalid CID would throw out_of_range.
  if (fWCEnabled && fWCPosition) {
    std::vector<std::vector<float>> wcWaves;
    wcWaves.reserve(3);
    wcWaves.push_back(anEvent.GetData(fCID_WCX).pedcorrectedWaveform());
    wcWaves.push_back(anEvent.GetData(fCID_WCY).pedcorrectedWaveform());
    wcWaves.push_back(anEvent.GetData(fCID_NIM).pedcorrectedWaveform());

    const auto posVec = GetPosition(wcWaves);
    if (posVec.size() == 2)
      fWCPosition->Fill(posVec.at(0), posVec.at(1));
  }

  // ── Hodoscope (16x16 IntADC and PeakADC maxima) ─────────────────────────
  if (fHodoEnabled)
    FillHodoscope(anEvent);
}

std::vector<float> TBaux::GetHodoscopeRawPosition(TBevt<TBwaveform> anEvent) {

  if (!fHodoEnabled) return {};
  if (fCID_HodoX.size() != 16 || fCID_HodoY.size() != 16) return {};

  // For each event, find the brightest fiber along X and the brightest
  // fiber along Y (separately for IntADC and PeakADC). Mirrors
  // draw_hodoscope.cc::247-266. Uses TBaux's own GetIntADC / GetPeakADC
  // (same pedestal-corrected integration and max-search as function.h's
  // GetInt / GetPeak, just member functions so we don't have to include
  // function.h — which would create duplicate-symbol linker errors
  // against the standalone draw_*.cc executables that already include it).
  std::vector<float> intADC_X(16, 0.f);
  std::vector<float> intADC_Y(16, 0.f);
  std::vector<float> peakADC_X(16, 0.f);
  std::vector<float> peakADC_Y(16, 0.f);

  for (int i = 0; i < 16; ++i) {
    const std::vector<short> wfX = anEvent.GetData(fCID_HodoX[i]).waveform();
    const std::vector<short> wfY = anEvent.GetData(fCID_HodoY[i]).waveform();
    if (wfX.size() > static_cast<size_t>(fHodoLastX[i])) {
      intADC_X[i]  = static_cast<float>(GetIntADC (wfX, fHodoFirstX[i], fHodoLastX[i]));
      peakADC_X[i] = static_cast<float>(GetPeakADC(wfX, fHodoFirstX[i], fHodoLastX[i]));
    }
    if (wfY.size() > static_cast<size_t>(fHodoLastY[i])) {
      intADC_Y[i]  = static_cast<float>(GetIntADC (wfY, fHodoFirstY[i], fHodoLastY[i]));
      peakADC_Y[i] = static_cast<float>(GetPeakADC(wfY, fHodoFirstY[i], fHodoLastY[i]));
    }
  }

  const int xIdxInt  = std::max_element(intADC_X.begin(),  intADC_X.end())  - intADC_X.begin();
  const int yIdxInt  = std::max_element(intADC_Y.begin(),  intADC_Y.end())  - intADC_Y.begin();
  const int xIdxPeak = std::max_element(peakADC_X.begin(), peakADC_X.end()) - peakADC_X.begin();
  const int yIdxPeak = std::max_element(peakADC_Y.begin(), peakADC_Y.end()) - peakADC_Y.begin();

  // Raw positions in fiber coordinates (bin centers, in [0.5, 15.5]).
  return {
    xIdxInt  + 0.5f, yIdxInt  + 0.5f,
    xIdxPeak + 0.5f, yIdxPeak + 0.5f,
  };
}

void TBaux::FillHodoscope(TBevt<TBwaveform> anEvent) {

  if (!fHodoIntADC || !fHodoPeakADC) return;

  const auto raw = GetHodoscopeRawPosition(anEvent);
  if (raw.size() != 4) return;

  const float rawX_int  = raw[0];
  const float rawY_int  = raw[1];
  const float rawX_peak = raw[2];
  const float rawY_peak = raw[3];

  // Center-corrected positions: shift by (-CENTER + nominal_center).
  // The nominal hodoscope center is (8, 8). When AUX.Hodoscope.CENTER is
  // left at the default [8, 8] this is the identity transform.
  const float corrX_int  = rawX_int  - fHodoCenter[0] + 8.0f;
  const float corrY_int  = rawY_int  - fHodoCenter[1] + 8.0f;
  const float corrX_peak = rawX_peak - fHodoCenter[0] + 8.0f;
  const float corrY_peak = rawY_peak - fHodoCenter[1] + 8.0f;

  fHodoIntADC      ->Fill(rawX_int,   rawY_int,   1);
  fHodoPeakADC     ->Fill(rawX_peak,  rawY_peak,  1);
  if (fHodoIntADC_corr ) fHodoIntADC_corr ->Fill(corrX_int,  corrY_int,  1);
  if (fHodoPeakADC_corr) fHodoPeakADC_corr->Fill(corrX_peak, corrY_peak, 1);
}

bool TBaux::IsPassing(TBevt<TBwaveform> anEvent) {

  // If the loaded mapping has no WC, we cannot compute the beam-spot or the
  // WC↔Hodo inclination, so --AUXcut is a no-op (let every event through).
  // The warning was printed once at init time.
  if (!fWCEnabled) return true;

  // ── (1) WC beam-spot cut (center-corrected; in mm, centered on 0) ──
  std::vector<std::vector<float>> wcWaves;
  wcWaves.reserve(3);
  wcWaves.push_back(anEvent.GetData(fCID_WCX).pedcorrectedWaveform());
  wcWaves.push_back(anEvent.GetData(fCID_WCY).pedcorrectedWaveform());
  wcWaves.push_back(anEvent.GetData(fCID_NIM).pedcorrectedWaveform());

  auto posVec = GetPosition(wcWaves); // X, Y in mm
  if (posVec.size() != 2)
    return false;

  if (fWCPosCut > 0) {
    if (std::abs(posVec.at(0)) > fWCPosCut)
      return false;
    if (std::abs(posVec.at(1)) > fWCPosCut)
      return false;
  }

  // ── (2) WC↔Hodo inclination cut (only in WCHodo mode) ────────────────
  // Both subsystems are expressed in mm relative to their own corrected
  // beam-center: WC's GetPosition() already returns mm centered on 0,
  // and we subtract fHodoCenter from the raw hodoscope bin centers to
  // bring the hodoscope into the same frame (1 fiber = 1 mm). A perfect
  // beam ends up at (0, 0) on both, so its difference is (0, 0) and
  // always passes; an inclined beam shows up as a non-zero difference.
  if (fAuxCutMode == "WCHodo") {
    const auto hodoRaw = GetHodoscopeRawPosition(anEvent);
    if (hodoRaw.size() != 4)
      return false;  // Hodoscope unavailable → fail-closed, like missing WC info.

    // GetHodoscopeRawPosition() returns { x_int, y_int, x_peak, y_peak }
    // in fiber units (= mm). AUX.Hodoscope.CUT_METHOD picks the pair fed
    // into the inclination cut; both metrics keep showing up in the AUX
    // plots regardless of this choice.
    const bool usePeak = (fHodoCutMethod == "PeakADC");
    const float hodo_x_raw = usePeak ? hodoRaw[2] : hodoRaw[0];
    const float hodo_y_raw = usePeak ? hodoRaw[3] : hodoRaw[1];
    const float hodo_x_centered = hodo_x_raw - static_cast<float>(fHodoCenter[0]);
    const float hodo_y_centered = hodo_y_raw - static_cast<float>(fHodoCenter[1]);

    const float dx = posVec.at(0) - hodo_x_centered;
    const float dy = posVec.at(1) - hodo_y_centered;

    if (std::abs(dx) > fInclinationCut[0]) return false;
    if (std::abs(dy) > fInclinationCut[1]) return false;
  }

  return true;
}

void TBaux::Draw() {

  if (!fCanvas || !fWCPosition)
    return;

  fCanvas->cd(1);
  fWCPosition->Draw("colz");

  gSystem->Sleep(1000);
}

void TBaux::SetMaximum() {

  // float max = -999;

  // if (fPS->GetMaximum() > max) max = fPS->GetMaximum();
  // if (fMC->GetMaximum() > max) max = fMC->GetMaximum();
  // if (fTC->GetMaximum() > max) max = fTC->GetMaximum();

  // fFrameTop->GetYaxis()->SetRangeUser(0., max * 1.2);


  // if (fPS->GetMaximum() > fMC->GetMaximum()) fFrameTop->GetYaxis()->SetRangeUser(0., fPS->GetMaximum() * 1.2);
  // else                                       fFrameTop->GetYaxis()->SetRangeUser(0., fMC->GetMaximum() * 1.2);

  // if (fCC1->GetMaximum() > fCC2->GetMaximum()) fFrameBot->GetYaxis()->SetRangeUser(0., fCC1->GetMaximum() * 1.2);
  // else                                         fFrameBot->GetYaxis()->SetRangeUser(0., fCC2->GetMaximum() * 1.2);
}

void TBaux::Update() {

  if (!fCanvas || !fWCPosition)
    return;

  if (fIsFirst) fIsFirst = false;

  SetMaximum();

  // ── Draw all AUX canvases ───────────────────────────────────────────────
  fCanvas->cd(1);
  fWCPosition->Draw("colz");
  fCanvas->cd();
  fCanvas->Update();
  if (fDraw) fCanvas->Pad()->Draw();

  if (fHodoEnabled && fCanvasHodoIntADC && fHodoIntADC) {
    fCanvasHodoIntADC->cd();
    fHodoIntADC->Draw("colz");
    fCanvasHodoIntADC->Update();
    if (fDraw) fCanvasHodoIntADC->Pad()->Draw();
  }

  if (fHodoEnabled && fCanvasHodoPeakADC && fHodoPeakADC) {
    fCanvasHodoPeakADC->cd();
    fHodoPeakADC->Draw("colz");
    fCanvasHodoPeakADC->Update();
    if (fDraw) fCanvasHodoPeakADC->Pad()->Draw();
  }

  if (fHodoEnabled && fCanvasHodoIntADC_corr && fHodoIntADC_corr) {
    fCanvasHodoIntADC_corr->cd();
    fHodoIntADC_corr->Draw("colz");
    fCanvasHodoIntADC_corr->Update();
    if (fDraw) fCanvasHodoIntADC_corr->Pad()->Draw();
  }

  if (fHodoEnabled && fCanvasHodoPeakADC_corr && fHodoPeakADC_corr) {
    fCanvasHodoPeakADC_corr->cd();
    fHodoPeakADC_corr->Draw("colz");
    fCanvasHodoPeakADC_corr->Update();
    if (fDraw) fCanvasHodoPeakADC_corr->Pad()->Draw();
  }

  // ── Combined AUX ROOT file (WC + hodoscope, raw and corrected) ─────────
  TString output = "./output/Run" + std::to_string(fRunNum) + "_AUX.root";
  if (fAuxCut) output = "./output/Run" + std::to_string(fRunNum) + "_AUX_AuxCut.root";
  {
    TFile outoutFile(output, "RECREATE");
    outoutFile.cd();
    fCanvas->Write();
    fWCPosition->Write();
    if (fHodoEnabled) {
      if (fCanvasHodoIntADC)       fCanvasHodoIntADC      ->Write();
      if (fCanvasHodoPeakADC)      fCanvasHodoPeakADC     ->Write();
      if (fCanvasHodoIntADC_corr)  fCanvasHodoIntADC_corr ->Write();
      if (fCanvasHodoPeakADC_corr) fCanvasHodoPeakADC_corr->Write();
      if (fHodoIntADC)             fHodoIntADC      ->Write();
      if (fHodoPeakADC)            fHodoPeakADC     ->Write();
      if (fHodoIntADC_corr)        fHodoIntADC_corr ->Write();
      if (fHodoPeakADC_corr)       fHodoPeakADC_corr->Write();
    }
    outoutFile.Close();
  }

  // ── Per-canvas JSON dumps ───────────────────────────────────────────────
  // The web run-browser scans Run<N>_<type>_<method>[_AuxCut]_<canvas>.json.
  // We split the AUX output across pseudo-"methods" so they show up as
  // distinguishable canvases in the run group:
  //   method=WC         → fCanvas_WC                (wire-chamber position)
  //   method=Hodoscope  → fCanvas_HodoIntADC        (raw 16x16, IntADC)
  //   method=Hodoscope  → fCanvas_HodoPeakADC       (raw 16x16, PeakADC)
  //   method=Hodoscope  → fCanvas_HodoIntADC_corr   (center-corrected, IntADC)
  //   method=Hodoscope  → fCanvas_HodoPeakADC_corr  (center-corrected, PeakADC)
  // Each write is atomic (.tmp -> rename) so a polling LIVE viewer never
  // reads a half-written file.
  auto dumpJSON = [&](TCanvas* canvas, const std::string& methodPart) {
    if (!canvas) return;
    std::string basePrefix = "Run" + std::to_string(fRunNum) + "_AUX_" + methodPart;
    if (fAuxCut) basePrefix += "_AuxCut";
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
  };

  dumpJSON(fCanvas, "WC");
  if (fHodoEnabled) {
    dumpJSON(fCanvasHodoIntADC,       "Hodoscope");
    dumpJSON(fCanvasHodoPeakADC,      "Hodoscope");
    dumpJSON(fCanvasHodoIntADC_corr,  "Hodoscope");
    dumpJSON(fCanvasHodoPeakADC_corr, "Hodoscope");
  }

  // Process pending GUI events only when the canvas is actually being
  // displayed (--DRAW). In batch mode (web UI / scripted runs) we must
  // NOT call fApp->Run(false): the TApplication is constructed without
  // SetReturnFromRun(true) for non-LIVE, so Run(false) would block here
  // forever — even though all output files have already been written.
  // This mirrors the pattern used in TBplotengine::Update().
  if (fDraw) gSystem->ProcessEvents();

  gSystem->Sleep(1000);
}

void TBaux::SaveAs(TString output) {

  if (output == "")
    output = "./output/Run" + std::to_string(fRunNum) + "_AUX.root";

  if (!fWCPosition)
    return;

  TFile* outoutFile = new TFile(output, "RECREATE");
  outoutFile->cd();

  fWCPosition->Write();

  outoutFile->Close();
}
