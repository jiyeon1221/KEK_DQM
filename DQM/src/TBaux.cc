#include "TBaux.h"
#include "GuiTypes.h"
#include "TSystem.h"
#include "TStyle.h"
#include "TBufferJSON.h"
#include <sys/types.h>
#include <fstream>
#include <cstdio>
#include <sstream>
#include <cctype>


TBaux::TBaux(const YAML::Node fNodePlot_, int fRunNum_, bool fPlotting_, bool fLive_, bool fDraw_, TButility fUtility_)
: fNodeAux(fNodePlot_),
  fRunNum(fRunNum_),
  fPlotting(fPlotting_),
  fLive(fLive_),
  fDraw(fDraw_),
  fAuxCut(false),
  fAuxCutMode("WC"),
  fAuxMode("WCHodo"),
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
  fDWCEnabled(false),
  fCID_DWC1R(), fCID_DWC1L(), fCID_DWC1U(), fCID_DWC1D(),
  fCID_DWC2R(), fCID_DWC2L(), fCID_DWC2U(), fCID_DWC2D(),
  fDWCThreshold(0.4),
  fDWCCalib({0.180654, 0.217961, -0.180342, -0.0994697, 0.181416, -0.00911072, -0.17822, -0.0489771}),
  fDWCCenter({-5.24, -8.416, -2.834, -9.524}),
  fDWCCorr(4.0),
  fDWCPosCut(-1.),
  fDWC1Pos(nullptr),
  fDWC2Pos(nullptr),
  fDWCXCorr(nullptr),
  fDWCYCorr(nullptr),
  fCanvasDWC(nullptr),
  fPIDEnabled(false),
  fCID_PS(), fCID_MC(), fCID_TC(), fCID_CC1(), fCID_CC2(),
  fPS(nullptr),
  fMC(nullptr),
  fTC(nullptr),
  fCC1(nullptr),
  fCC2(nullptr),
  fCanvasPID(nullptr),
  fPSThreshold(424.0),
  fMCVetoThreshold(38.0),
  fCC1cut(-1.),
  fCC2cut(-1.),
  fPSInitCut(-1.),
  fPSFinCut(1.0e9),
  fHodoEnabled(false),
  fCID_HodoX(),
  fCID_HodoY(),
  fHodoFirstX(16, 150),
  fHodoLastX(16, 350),
  fHodoFirstY(16, 150),
  fHodoLastY(16, 350),
  fHodoCenter(2, 8.0f),
  fHodoCutMethod("IntADC"),
  fHodoNormIntADC_X(16, 1.0),
  fHodoNormIntADC_Y(16, 1.0),
  fHodoNormPeakADC_X(16, 1.0),
  fHodoNormPeakADC_Y(16, 1.0),
  fHodoIntADC(nullptr),
  fHodoPeakADC(nullptr),
  fCanvasHodoIntADC(nullptr),
  fCanvasHodoPeakADC(nullptr),
  fHodoIntADC_corr(nullptr),
  fHodoPeakADC_corr(nullptr),
  fCanvasHodoIntADC_corr(nullptr),
  fCanvasHodoPeakADC_corr(nullptr),
  fSqHodoEnabled(false),
  fCID_SqHodoX(), fCID_SqHodoY(),
  fSqHodoFirst(150), fSqHodoLast(350),
  fSqHodoCenter({14.5, 14.5}),
  fSqHodoNormIntADC_X(29, 1.0), fSqHodoNormIntADC_Y(29, 1.0),
  fSqHodoNormPeakADC_X(29, 1.0), fSqHodoNormPeakADC_Y(29, 1.0),
  fSqHodoThrIntADC_X(29, 0.0), fSqHodoThrIntADC_Y(29, 0.0),
  fSqHodoThrPeakADC_X(29, 0.0), fSqHodoThrPeakADC_Y(29, 0.0),
  fSqHodoIntADC(nullptr),
  fSqHodoPeakADC(nullptr),
  fCanvasSqHodoIntADC(nullptr),
  fCanvasSqHodoPeakADC(nullptr),
  fRndHodoEnabled(false),
  fCID_RndHodoX(), fCID_RndHodoY(),
  fRndHodoFirst(150), fRndHodoLast(350),
  fRndHodoCenter({8.25, 8.25}),
  fRndHodoNormIntADC_X(32, 1.0), fRndHodoNormIntADC_Y(32, 1.0),
  fRndHodoNormPeakADC_X(32, 1.0), fRndHodoNormPeakADC_Y(32, 1.0),
  fRndHodoThrIntADC_X(32, 0.0), fRndHodoThrIntADC_Y(32, 0.0),
  fRndHodoThrPeakADC_X(32, 0.0), fRndHodoThrPeakADC_Y(32, 0.0),
  fRndHodoIntADC(nullptr),
  fRndHodoPeakADC(nullptr),
  fCanvasRndHodoIntADC(nullptr),
  fCanvasRndHodoPeakADC(nullptr)
{}

void TBaux::init() {

  // AUX.WC is only mandatory in YAML when the operator actually asked for
  // WC (via --AUXMode or a WC-based --AUXCutMode). WC is currently hidden
  // from the web UI (CERN has no wire chamber), so a config without an
  // AUX.WC block must not block DWC/PID-only runs; the strict validation
  // below only runs once we know WC is actually requested (see needWC
  // computed just below — this whole block is re-entered defensively so
  // config typos still fail loudly when WC *is* requested).
  const auto nodeWC = fNodeAux["WC"];

  // ── Decide which AUX subsystems we actually need to load ───────────────
  // The operator picks --AUXMode as a comma-separated list of any of
  // "WC", "Hodo", "DWC", "PID" (or the legacy single tokens "WC" | "Hodo"
  // | "WCHodo", where "WCHodo" expands to "WC,Hodo"). Separately --AUXcut
  // may require specific subsystems depending on --AUXCutMode: "WC" needs
  // WC, "WCHodo" needs WC+Hodo, "DWC" needs DWC, "DWCPID" needs DWC+PID.
  // We "need" a subsystem if EITHER the plot path OR the cut path asks
  // for it. This is the gate that prevents TBread from trying to open a
  // MID for a subsystem that is physically out of the setup (hodoscope,
  // or — at CERN — the wire chamber) when the operator hasn't asked for
  // it.
  bool modeWantsWC = false, modeWantsHodo = false, modeWantsDWC = false, modeWantsPID = false;
  {
    std::vector<std::string> tokens;
    std::istringstream iss(fAuxMode);
    std::string tok;
    while (std::getline(iss, tok, ',')) tokens.push_back(tok);

    bool anyRecognised = false;
    for (std::string tok2 : tokens) {
      while (!tok2.empty() && std::isspace((unsigned char)tok2.front())) tok2.erase(tok2.begin());
      while (!tok2.empty() && std::isspace((unsigned char)tok2.back()))  tok2.pop_back();

      if (tok2 == "WCHodo")     { modeWantsWC = true; modeWantsHodo = true; anyRecognised = true; }
      else if (tok2 == "WC")   { modeWantsWC = true;   anyRecognised = true; }
      else if (tok2 == "Hodo") { modeWantsHodo = true; anyRecognised = true; }
      else if (tok2 == "DWC")  { modeWantsDWC = true;  anyRecognised = true; }
      else if (tok2 == "PID")  { modeWantsPID = true;  anyRecognised = true; }
      else if (!tok2.empty()) {
        std::cout << "[TBaux] Unrecognised --AUXMode token '" << tok2 << "', ignoring." << std::endl;
      }
    }
    if (!anyRecognised) {
      std::cout << "[TBaux] No valid --AUXMode tokens found in '" << fAuxMode
                << "', falling back to 'WC,Hodo'." << std::endl;
      modeWantsWC = true;
      modeWantsHodo = true;
    }
  }

  const bool needWC   = (fPlotting && modeWantsWC)   || (fAuxCut && (fAuxCutMode == "WC" || fAuxCutMode == "WCHodo"));
  const bool needHodo = (fPlotting && modeWantsHodo) || (fAuxCut && fAuxCutMode == "WCHodo");
  const bool needDWC  = (fPlotting && modeWantsDWC)  || (fAuxCut && (fAuxCutMode == "DWC" || fAuxCutMode == "DWCPID"));
  const bool needPID  = (fPlotting && modeWantsPID)  || (fAuxCut && (fAuxCutMode == "DWCPID" || fAuxCutMode == "PID"));

  std::cout << "[TBaux] AUXMode='" << fAuxMode
            << "' AUXcut=" << (fAuxCut ? "on" : "off")
            << " AUXCutMode='" << fAuxCutMode << "'"
            << " → load WC=" << (needWC ? "yes" : "no")
            << ", load Hodo=" << (needHodo ? "yes" : "no")
            << ", load DWC=" << (needDWC ? "yes" : "no")
            << ", load PID=" << (needPID ? "yes" : "no")
            << std::endl;

  // ── WC config (only mandatory when WC is actually needed) ──────────────
  // Strict validation (throws on missing/malformed YAML) only runs here,
  // gated on needWC, so a DWC/PID-only config (no AUX.WC block at all —
  // the expected CERN setup, no wire chamber) never trips this.
  if (needWC) {
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
    if (nodeWC["THRESHOLD"]) fWCThreshold = nodeWC["THRESHOLD"].as<double>();
    if (nodeWC["POSCUT"])    fWCPosCut    = nodeWC["POSCUT"].as<double>();
  }

  // ── WC channel resolution (only if needed) ─────────────────────────────
  // TButility::GetCID() returns TBcid(-1, -1) when the channel name is not
  // present in the loaded mapping. Defer enabling WC processing until we
  // know every WC channel resolved — otherwise Fill() and IsPassing() will
  // call anEvent.GetData(TBcid(-1, -1)).waveform() and throw out_of_range.
  auto cidValid = [](const TBcid& c) { return c.mid() >= 0 && c.channel() >= 0; };

  fCIDtoPlot.clear();
  if (needWC) {
    fCID_WCX = fUtility.GetCID("WCX");
    fCID_WCY = fUtility.GetCID("WCY");
    fCID_NIM = fUtility.GetCID("NIM");
    fWCEnabled = cidValid(fCID_WCX) && cidValid(fCID_WCY) && cidValid(fCID_NIM);
    if (!fWCEnabled) {
      std::cout << "[TBaux] WC channels missing from the loaded mapping "
                << "(WCX/WCY/NIM); --AUX plots and --AUXcut will be skipped."
                << std::endl;
    } else {
      // Only push CIDs into the reader's MID list when they actually
      // resolved — pushing TBcid(-1,-1) would still produce MID -1
      // which GetUniqueMID filters, but being explicit is safer.
      fCIDtoPlot.push_back(fCID_WCX);
      fCIDtoPlot.push_back(fCID_WCY);
      fCIDtoPlot.push_back(fCID_NIM);
    }
  } else {
    // WC not requested — leave fWCEnabled=false (the constructor default)
    // so Fill() / IsPassing() short-circuit cleanly without trying to
    // read WC waveforms that we never told the reader to load.
    fWCEnabled = false;
    std::cout << "[TBaux] WC plots/cut not requested (AUXMode='" << fAuxMode
              << "', AUXcut=" << (fAuxCut ? "on" : "off") << ")"
              << " — skipping WC channel load." << std::endl;
  }

  // WC histogram + canvas — only allocated when WC is actually needed,
  // so a Hodo-only run doesn't leave an empty "WC_Position" ROOT object
  // or a stray fCanvas_WC JSON dump in the output dir.
  if (needWC) {
    fWCPosition = new TH2D(
      "WC_Position",
      (TString)"Run " + std::to_string(fRunNum) + " Wire Chamber position;X [mm];Y [mm]",
      120, -30., 30., 120, -30., 30.);
    fWCPosition->SetStats(0);

    fCanvas = new TCanvas("fCanvas_WC", "fCanvas_WC", 1200, 800);
    fCanvas->Divide(1, 1);
    fCanvas->cd(1)->SetRightMargin(0.13);
  }

  // ── DWC channel resolution (only if needed) ─────────────────────────────
  // Restored from TB2025. Mirrors the WC pattern above: resolve all 8
  // timing-edge CIDs, only enable + push into fCIDtoPlot if every one of
  // them is present in the loaded mapping (e.g. the DWC boxes might not be
  // installed for a given run).
  if (needDWC) {
    fCID_DWC1R = fUtility.GetCID("DWC1R");
    fCID_DWC1L = fUtility.GetCID("DWC1L");
    fCID_DWC1U = fUtility.GetCID("DWC1U");
    fCID_DWC1D = fUtility.GetCID("DWC1D");
    fCID_DWC2R = fUtility.GetCID("DWC2R");
    fCID_DWC2L = fUtility.GetCID("DWC2L");
    fCID_DWC2U = fUtility.GetCID("DWC2U");
    fCID_DWC2D = fUtility.GetCID("DWC2D");
    fDWCEnabled = cidValid(fCID_DWC1R) && cidValid(fCID_DWC1L) && cidValid(fCID_DWC1U) && cidValid(fCID_DWC1D)
               && cidValid(fCID_DWC2R) && cidValid(fCID_DWC2L) && cidValid(fCID_DWC2U) && cidValid(fCID_DWC2D);
    if (!fDWCEnabled) {
      std::cout << "[TBaux] DWC channels missing from the loaded mapping "
                << "(DWC1R/L/U/D, DWC2R/L/U/D); DWC plots and DWC-based "
                << "--AUXcut modes will be skipped." << std::endl;
    } else {
      fCIDtoPlot.push_back(fCID_DWC1R); fCIDtoPlot.push_back(fCID_DWC1L);
      fCIDtoPlot.push_back(fCID_DWC1U); fCIDtoPlot.push_back(fCID_DWC1D);
      fCIDtoPlot.push_back(fCID_DWC2R); fCIDtoPlot.push_back(fCID_DWC2L);
      fCIDtoPlot.push_back(fCID_DWC2U); fCIDtoPlot.push_back(fCID_DWC2D);

      const auto nodeDWC = fNodeAux["DWC"];
      if (nodeDWC) {
        if (nodeDWC["THRESHOLD"]) fDWCThreshold = nodeDWC["THRESHOLD"].as<double>();
        if (nodeDWC["CALIB"]) {
          const auto v = nodeDWC["CALIB"].as<std::vector<double>>();
          if (v.size() == 8) fDWCCalib = v;
          else std::cout << "[TBaux] AUX.DWC.CALIB must have exactly 8 entries; keeping defaults." << std::endl;
        }
        if (nodeDWC["CENTER"]) {
          const auto v = nodeDWC["CENTER"].as<std::vector<double>>();
          if (v.size() == 4) fDWCCenter = v;
          else std::cout << "[TBaux] AUX.DWC.CENTER must have exactly 4 entries; keeping defaults." << std::endl;
        }
        if (nodeDWC["CORR"])   fDWCCorr   = nodeDWC["CORR"].as<double>();
        if (nodeDWC["POSCUT"]) fDWCPosCut = nodeDWC["POSCUT"].as<double>();
      } else {
        std::cout << "[TBaux] AUX.DWC is not configured in YAML; using built-in "
                  << "placeholder calibration constants (see TBaux.h)." << std::endl;
      }
    }
  } else {
    fDWCEnabled = false;
    std::cout << "[TBaux] DWC plots/cut not requested (AUXMode='" << fAuxMode
              << "', AUXCutMode='" << fAuxCutMode << "')"
              << " — skipping DWC channel load." << std::endl;
  }

  if (needDWC) {
    fDWC1Pos = new TH2D(
      "DWC1_Position",
      (TString)"Run " + std::to_string(fRunNum) + " DWC1 position;X [mm];Y [mm]",
      120, -30., 30., 120, -30., 30.);
    fDWC1Pos->SetStats(0);

    fDWC2Pos = new TH2D(
      "DWC2_Position",
      (TString)"Run " + std::to_string(fRunNum) + " DWC2 position;X [mm];Y [mm]",
      120, -30., 30., 120, -30., 30.);
    fDWC2Pos->SetStats(0);

    fDWCXCorr = new TH2D(
      "DWC_XCorrelation",
      (TString)"Run " + std::to_string(fRunNum) + " DWC1 X vs DWC2 X;DWC1 X [mm];DWC2 X [mm]",
      120, -30., 30., 120, -30., 30.);
    fDWCXCorr->SetStats(0);

    fDWCYCorr = new TH2D(
      "DWC_YCorrelation",
      (TString)"Run " + std::to_string(fRunNum) + " DWC1 Y vs DWC2 Y;DWC1 Y [mm];DWC2 Y [mm]",
      120, -30., 30., 120, -30., 30.);
    fDWCYCorr->SetStats(0);

    fCanvasDWC = new TCanvas("fCanvas_DWC", "fCanvas_DWC", 1600, 1600);
    fCanvasDWC->Divide(2, 2);
    for (int p = 1; p <= 4; ++p) fCanvasDWC->cd(p)->SetRightMargin(0.13);
  }

  // ── PID channel resolution (PS, MC, TC, CC1, CC2; only if needed) ──────
  // Restored from TB2025.
  if (needPID) {
    fCID_PS  = fUtility.GetCID("PS");
    fCID_MC  = fUtility.GetCID("MC");
    fCID_TC  = fUtility.GetCID("TC");
    fCID_CC1 = fUtility.GetCID("CC1");
    fCID_CC2 = fUtility.GetCID("CC2");
    fPIDEnabled = cidValid(fCID_PS) && cidValid(fCID_MC) && cidValid(fCID_TC)
               && cidValid(fCID_CC1) && cidValid(fCID_CC2);
    if (!fPIDEnabled) {
      std::cout << "[TBaux] PID channels missing from the loaded mapping "
                << "(PS/MC/TC/CC1/CC2); PID plots and the DWCPID --AUXcut mode "
                << "will be skipped." << std::endl;
    } else {
      fCIDtoPlot.push_back(fCID_PS);
      fCIDtoPlot.push_back(fCID_MC);
      fCIDtoPlot.push_back(fCID_TC);
      fCIDtoPlot.push_back(fCID_CC1);
      fCIDtoPlot.push_back(fCID_CC2);

      const auto nodePID = fNodeAux["PID"];
      if (nodePID) {
        if (nodePID["PS_THRESHOLD_PEAKADC"]) fPSThreshold     = nodePID["PS_THRESHOLD_PEAKADC"].as<double>();
        if (nodePID["MC_VETO_PEAKADC"])      fMCVetoThreshold = nodePID["MC_VETO_PEAKADC"].as<double>();
      } else {
        std::cout << "[TBaux] AUX.PID is not configured in YAML; using built-in "
                  << "placeholder thresholds (PS >= " << fPSThreshold
                  << ", MC >= " << fMCVetoThreshold << ")." << std::endl;
      }
    }
  } else {
    fPIDEnabled = false;
    std::cout << "[TBaux] PID plots/cut not requested (AUXMode='" << fAuxMode
              << "', AUXCutMode='" << fAuxCutMode << "')"
              << " — skipping PID channel load." << std::endl;
  }

  if (needPID) {
    const int psBins = 1152; // PeakADC binning, matches TBplotengine's convention
    fPS  = new TH1D("PS",  (TString)"Run " + std::to_string(fRunNum) + " PS;PeakADC;nEvents",       psBins, -512., 4096.);
    fMC  = new TH1D("MC",  (TString)"Run " + std::to_string(fRunNum) + " MC;PeakADC;nEvents",       psBins, -512., 4096.);
    fTC  = new TH1D("TC",  (TString)"Run " + std::to_string(fRunNum) + " TC;PeakADC;nEvents",       psBins, -512., 4096.);
    fCC1 = new TH1D("CC1", (TString)"Run " + std::to_string(fRunNum) + " CC1 + CC2;PeakADC;nEvents", psBins, -512., 4096.);
    fCC2 = new TH1D("CC2", (TString)"Run " + std::to_string(fRunNum) + " CC2;PeakADC;nEvents",       psBins, -512., 4096.);
    fPS->SetStats(1);  fPS->SetLineColor(kBlue + 1);   fPS->SetLineWidth(2);
    fMC->SetStats(1);  fMC->SetLineColor(kRed + 1);    fMC->SetLineWidth(2);
    fTC->SetStats(1);  fTC->SetLineColor(kGreen + 2);  fTC->SetLineWidth(2);
    fCC1->SetStats(1); fCC1->SetLineColor(kAzure + 2); fCC1->SetLineWidth(2);
    fCC2->SetStats(1); fCC2->SetLineColor(kOrange + 7);fCC2->SetLineWidth(2);

    fCanvasPID = new TCanvas("fCanvas_PID", "fCanvas_PID", 1600, 1600);
    fCanvasPID->Divide(2, 2);
  }

  // ── Read Hodo YAML config (cheap; always done regardless of needHodo) ─
  // We pull AUX.Hodoscope.CENTER / CUT_METHOD / NORM_CONST_* unconditionally
  // because the YAML reads are essentially free and keep the helpers
  // (e.g. center correction) populated even if a later code path enables
  // Hodo. Channel resolution + canvas creation, which are the actually
  // expensive (and MID-17-loading) parts, stay gated below.
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

  // Per-channel normalization constants from config_general.yml.
  // Each vector has 16 entries [HX1..HX16] or [HY1..HY16].
  // Values <= 0 are replaced with 1.0 to avoid division by zero.
  auto loadNormConst = [](const YAML::Node& node, std::vector<double>& dst) {
    if (!node) return;
    const auto v = node.as<std::vector<double>>();
    if (v.size() != 16) return;
    for (int i = 0; i < 16; ++i)
      dst[i] = (v[i] > 1e-9) ? v[i] : 1.0;
  };

  if (nodeHodo && nodeHodo["NORM_CONST_INTADC"]) {
    loadNormConst(nodeHodo["NORM_CONST_INTADC"]["HX"], fHodoNormIntADC_X);
    loadNormConst(nodeHodo["NORM_CONST_INTADC"]["HY"], fHodoNormIntADC_Y);
  }
  if (nodeHodo && nodeHodo["NORM_CONST_PEAKADC"]) {
    loadNormConst(nodeHodo["NORM_CONST_PEAKADC"]["HX"], fHodoNormPeakADC_X);
    loadNormConst(nodeHodo["NORM_CONST_PEAKADC"]["HY"], fHodoNormPeakADC_Y);
  }

  // Inclination cut [X, Y] in mm, applied only in --AUXCutMode WCHodo.
  // Kept at AUX-top-level because the cut spans both subsystems.
  if (fNodeAux["INCLINATION_CUT"]) {
    const auto v = fNodeAux["INCLINATION_CUT"].as<std::vector<double>>();
    if (v.size() == 2) fInclinationCut = v;
  }

  // ── CERN square/round hodoscope cheap YAML reads ────────────────────────
  // Same "always read, gate channel resolution on needHodo" pattern as the
  // legacy Hodo block above. See AUX.Hodoscope.SQUARE / AUX.Hodoscope.ROUND
  // in config_general.yml.
  auto loadNormVecN = [](const YAML::Node& node, std::vector<double>& dst) {
    if (!node) return;
    const auto v = node.as<std::vector<double>>();
    if (v.size() != dst.size()) return;
    for (size_t i = 0; i < dst.size(); ++i)
      dst[i] = (v[i] > 1e-9) ? v[i] : 1.0;
  };
  auto loadThrVecN = [](const YAML::Node& node, std::vector<double>& dst) {
    if (!node) return;
    const auto v = node.as<std::vector<double>>();
    if (v.size() != dst.size()) return;
    dst = v;
  };
  // Concatenates two 16-entry layer arrays (layer1, layer2) into the
  // 32-entry vectors used by fCID_RndHodo{X,Y} ([layer1 0..15, layer2 16..31]).
  auto loadNormVecRound = [](const YAML::Node& layer1, const YAML::Node& layer2, std::vector<double>& dst) {
    if (dst.size() != 32) return;
    if (layer1) {
      const auto v = layer1.as<std::vector<double>>();
      if (v.size() == 16) for (int i = 0; i < 16; ++i) dst[i] = (v[i] > 1e-9) ? v[i] : 1.0;
    }
    if (layer2) {
      const auto v = layer2.as<std::vector<double>>();
      if (v.size() == 16) for (int i = 0; i < 16; ++i) dst[16 + i] = (v[i] > 1e-9) ? v[i] : 1.0;
    }
  };
  auto loadThrVecRound = [](const YAML::Node& layer1, const YAML::Node& layer2, std::vector<double>& dst) {
    if (dst.size() != 32) return;
    if (layer1) {
      const auto v = layer1.as<std::vector<double>>();
      if (v.size() == 16) for (int i = 0; i < 16; ++i) dst[i] = v[i];
    }
    if (layer2) {
      const auto v = layer2.as<std::vector<double>>();
      if (v.size() == 16) for (int i = 0; i < 16; ++i) dst[16 + i] = v[i];
    }
  };

  const auto nodeSqHodo = nodeHodo ? nodeHodo["SQUARE"] : YAML::Node();
  if (nodeSqHodo) {
    if (nodeSqHodo["CENTER"]) {
      const auto c = nodeSqHodo["CENTER"].as<std::vector<double>>();
      if (c.size() == 2) fSqHodoCenter = c;
    }
    if (nodeSqHodo["RANGE"]) {
      const auto r = nodeSqHodo["RANGE"].as<std::vector<int>>();
      if (r.size() == 2) { fSqHodoFirst = r[0]; fSqHodoLast = r[1]; }
    }
    if (nodeSqHodo["NORM_CONST_INTADC"]) {
      loadNormVecN(nodeSqHodo["NORM_CONST_INTADC"]["SHX"], fSqHodoNormIntADC_X);
      loadNormVecN(nodeSqHodo["NORM_CONST_INTADC"]["SHY"], fSqHodoNormIntADC_Y);
    }
    if (nodeSqHodo["NORM_CONST_PEAKADC"]) {
      loadNormVecN(nodeSqHodo["NORM_CONST_PEAKADC"]["SHX"], fSqHodoNormPeakADC_X);
      loadNormVecN(nodeSqHodo["NORM_CONST_PEAKADC"]["SHY"], fSqHodoNormPeakADC_Y);
    }
    if (nodeSqHodo["THRESHOLD_INTADC"]) {
      loadThrVecN(nodeSqHodo["THRESHOLD_INTADC"]["SHX"], fSqHodoThrIntADC_X);
      loadThrVecN(nodeSqHodo["THRESHOLD_INTADC"]["SHY"], fSqHodoThrIntADC_Y);
    }
    if (nodeSqHodo["THRESHOLD_PEAKADC"]) {
      loadThrVecN(nodeSqHodo["THRESHOLD_PEAKADC"]["SHX"], fSqHodoThrPeakADC_X);
      loadThrVecN(nodeSqHodo["THRESHOLD_PEAKADC"]["SHY"], fSqHodoThrPeakADC_Y);
    }
  }

  const auto nodeRndHodo = nodeHodo ? nodeHodo["ROUND"] : YAML::Node();
  if (nodeRndHodo) {
    if (nodeRndHodo["CENTER"]) {
      const auto c = nodeRndHodo["CENTER"].as<std::vector<double>>();
      if (c.size() == 2) fRndHodoCenter = c;
    }
    if (nodeRndHodo["RANGE"]) {
      const auto r = nodeRndHodo["RANGE"].as<std::vector<int>>();
      if (r.size() == 2) { fRndHodoFirst = r[0]; fRndHodoLast = r[1]; }
    }
    if (nodeRndHodo["NORM_CONST_INTADC"]) {
      loadNormVecRound(nodeRndHodo["NORM_CONST_INTADC"]["RH1X"], nodeRndHodo["NORM_CONST_INTADC"]["RH2X"], fRndHodoNormIntADC_X);
      loadNormVecRound(nodeRndHodo["NORM_CONST_INTADC"]["RH1Y"], nodeRndHodo["NORM_CONST_INTADC"]["RH2Y"], fRndHodoNormIntADC_Y);
    }
    if (nodeRndHodo["NORM_CONST_PEAKADC"]) {
      loadNormVecRound(nodeRndHodo["NORM_CONST_PEAKADC"]["RH1X"], nodeRndHodo["NORM_CONST_PEAKADC"]["RH2X"], fRndHodoNormPeakADC_X);
      loadNormVecRound(nodeRndHodo["NORM_CONST_PEAKADC"]["RH1Y"], nodeRndHodo["NORM_CONST_PEAKADC"]["RH2Y"], fRndHodoNormPeakADC_Y);
    }
    if (nodeRndHodo["THRESHOLD_INTADC"]) {
      loadThrVecRound(nodeRndHodo["THRESHOLD_INTADC"]["RH1X"], nodeRndHodo["THRESHOLD_INTADC"]["RH2X"], fRndHodoThrIntADC_X);
      loadThrVecRound(nodeRndHodo["THRESHOLD_INTADC"]["RH1Y"], nodeRndHodo["THRESHOLD_INTADC"]["RH2Y"], fRndHodoThrIntADC_Y);
    }
    if (nodeRndHodo["THRESHOLD_PEAKADC"]) {
      loadThrVecRound(nodeRndHodo["THRESHOLD_PEAKADC"]["RH1X"], nodeRndHodo["THRESHOLD_PEAKADC"]["RH2X"], fRndHodoThrPeakADC_X);
      loadThrVecRound(nodeRndHodo["THRESHOLD_PEAKADC"]["RH1Y"], nodeRndHodo["THRESHOLD_PEAKADC"]["RH2Y"], fRndHodoThrPeakADC_Y);
    }
  }

  // ── Hodoscope channel resolution + histograms (only if needed) ─────────
  // 16 X-fibers and 16 Y-fibers. Channel names must exist in the mapping
  // file pointed to by config_general.yml::Mapping (e.g. toymapping_v2.root
  // contains HX1..HX16 / HY1..HY16). The order here defines the fiber
  // index 0..15 used in the 2D hit-map plots — keep it strictly sequential
  // so bin N == fiber HX(N+1).
  //
  // When the operator selects --AUXMode WC (and --AUXCutMode != WCHodo),
  // this whole block is skipped: no CIDs resolved, no canvases created,
  // no MID-17 push into fCIDtoPlot, so TBread will never try to open the
  // hodoscope data files even if the mapping still lists HX/HY channels.
  if (needHodo) {
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
    } else {
      // Push CIDs only when the mapping is complete, so a partial
      // mapping doesn't accidentally send MID -1 / random partial MIDs
      // to TBread.
      for (const auto& cid : fCID_HodoX) fCIDtoPlot.push_back(cid);
      for (const auto& cid : fCID_HodoY) fCIDtoPlot.push_back(cid);
    }

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
  } else {
    std::cout << "[TBaux] Hodo plots/cut not requested (AUXMode='" << fAuxMode
              << "', AUXCutMode='" << fAuxCutMode << "')"
              << " — skipping hodoscope channel load (no MID 17 read)."
              << std::endl;
    fHodoEnabled = false;
  }

  // ── CERN square hodoscope channel resolution + histograms ───────────────
  // Gated on the same needHodo flag as the legacy 16x16 hodoscope: the web
  // UI has a single "Hodo" checkbox that turns on all hodoscope plotting.
  if (needHodo) {
    fCID_SqHodoX.clear();
    fCID_SqHodoY.clear();
    fCID_SqHodoX.reserve(29);
    fCID_SqHodoY.reserve(29);
    for (int i = 1; i <= 29; ++i) fCID_SqHodoX.push_back(fUtility.GetCID("SHX" + std::to_string(i)));
    for (int i = 1; i <= 29; ++i) fCID_SqHodoY.push_back(fUtility.GetCID("SHY" + std::to_string(i)));

    bool sqAllValid = true;
    for (const auto& c : fCID_SqHodoX) sqAllValid &= cidValid(c);
    for (const auto& c : fCID_SqHodoY) sqAllValid &= cidValid(c);
    fSqHodoEnabled = sqAllValid;

    if (!sqAllValid) {
      std::cout << "[TBaux] Square-hodoscope channels missing from the loaded "
                << "mapping (SHX1..29 / SHY1..29); square hodoscope AUX plot "
                << "will be skipped." << std::endl;
    } else {
      for (const auto& c : fCID_SqHodoX) fCIDtoPlot.push_back(c);
      for (const auto& c : fCID_SqHodoY) fCIDtoPlot.push_back(c);

      // Active area is 29x29 mm, fiber i covers [i-1, i) -> raw fiber
      // centers run 0.5..28.5. Histogram axis is fixed at [-14.5, +14.5]
      // (half of 29 mm); AUX.Hodoscope.SQUARE.CENTER only shifts which raw
      // position maps to 0 (default = active-area center = no shift).
      const double half = 14.5;
      fSqHodoIntADC = new TH2F(
        "hodoscope_sq_intADC",
        (TString)"Run " + std::to_string(fRunNum) + " Square Hodoscope IntADC;X [mm];Y [mm];events",
        29, -half, half, 29, -half, half);
      fSqHodoIntADC->SetStats(0);

      fSqHodoPeakADC = new TH2F(
        "hodoscope_sq_peakADC",
        (TString)"Run " + std::to_string(fRunNum) + " Square Hodoscope PeakADC;X [mm];Y [mm];events",
        29, -half, half, 29, -half, half);
      fSqHodoPeakADC->SetStats(0);

      fCanvasSqHodoIntADC = new TCanvas("fCanvas_SqHodoIntADC", "fCanvas_SqHodoIntADC", 800, 800);
      fCanvasSqHodoIntADC->cd()->SetRightMargin(0.13);

      fCanvasSqHodoPeakADC = new TCanvas("fCanvas_SqHodoPeakADC", "fCanvas_SqHodoPeakADC", 800, 800);
      fCanvasSqHodoPeakADC->cd()->SetRightMargin(0.13);
    }
  } else {
    fSqHodoEnabled = false;
  }

  // ── CERN round hodoscope channel resolution + histograms ────────────────
  if (needHodo) {
    fCID_RndHodoX.clear();
    fCID_RndHodoY.clear();
    fCID_RndHodoX.reserve(32);
    fCID_RndHodoY.reserve(32);
    // Fixed order: layer 1 fibers 1..16, then layer 2 fibers 1..16 — kept
    // consistent with GetRoundHodoRawPosition()'s index -> (layer, fiber)
    // decoding (idx<16 -> layer1, else layer2; fiber = idx%16 + 1).
    for (int layer = 1; layer <= 2; ++layer)
      for (int i = 1; i <= 16; ++i)
        fCID_RndHodoX.push_back(fUtility.GetCID("RH" + std::to_string(layer) + "X" + std::to_string(i)));
    for (int layer = 1; layer <= 2; ++layer)
      for (int i = 1; i <= 16; ++i)
        fCID_RndHodoY.push_back(fUtility.GetCID("RH" + std::to_string(layer) + "Y" + std::to_string(i)));

    bool rndAllValid = true;
    for (const auto& c : fCID_RndHodoX) rndAllValid &= cidValid(c);
    for (const auto& c : fCID_RndHodoY) rndAllValid &= cidValid(c);
    fRndHodoEnabled = rndAllValid;

    if (!rndAllValid) {
      std::cout << "[TBaux] Round-hodoscope channels missing from the loaded "
                << "mapping (RH1X/RH1Y/RH2X/RH2Y 1..16); round hodoscope AUX "
                << "plot will be skipped." << std::endl;
    } else {
      for (const auto& c : fCID_RndHodoX) fCIDtoPlot.push_back(c);
      for (const auto& c : fCID_RndHodoY) fCIDtoPlot.push_back(c);

      // Active area is 16.5x16.5 mm (layer 1 covers 0-16 mm, layer 2 is
      // offset by half a fiber and covers 0.5-16.5 mm), 0.5 mm bins.
      // Histogram axis is fixed at [-8.25, +8.25]; CENTER only shifts the
      // raw-position origin (default = active-area center = no shift).
      constexpr double kActive = 16.5;
      constexpr int nBins = 33;
      const double half = kActive / 2.0;

      fRndHodoIntADC = new TH2F(
        "hodoscope_rnd_intADC",
        (TString)"Run " + std::to_string(fRunNum) + " Round Hodoscope IntADC;X [mm];Y [mm];events",
        nBins, -half, half, nBins, -half, half);
      fRndHodoIntADC->SetStats(0);

      fRndHodoPeakADC = new TH2F(
        "hodoscope_rnd_peakADC",
        (TString)"Run " + std::to_string(fRunNum) + " Round Hodoscope PeakADC;X [mm];Y [mm];events",
        nBins, -half, half, nBins, -half, half);
      fRndHodoPeakADC->SetStats(0);

      fCanvasRndHodoIntADC = new TCanvas("fCanvas_RndHodoIntADC", "fCanvas_RndHodoIntADC", 800, 800);
      fCanvasRndHodoIntADC->cd()->SetRightMargin(0.13);

      fCanvasRndHodoPeakADC = new TCanvas("fCanvas_RndHodoPeakADC", "fCanvas_RndHodoPeakADC", 800, 800);
      fCanvasRndHodoPeakADC->cd()->SetRightMargin(0.13);
    }
  } else {
    fRndHodoEnabled = false;
  }
}

void TBaux::SetParticle(std::string fParticle_) {

  fParticle = fParticle_;

  // Restored from TB2025: AUX.<PARTICLE>.{CC1,CC2,PS_INIT,PS_FIN}, used by
  // the DWCPID --AUXcut mode (see IsPassing()). Any particle name is
  // accepted here (not just PION/KAON/PROTON) as long as config_general.yml
  // has a matching top-level AUX.<name> block; unknown names just print a
  // warning and leave the previous (or default) cut values untouched.
  const auto nodeParticle = fNodeAux[fParticle];
  if (!nodeParticle) {
    if (fParticle != "null" && !fParticle.empty()) {
      std::cout << "[TBaux] --particle '" << fParticle << "' has no AUX." << fParticle
                << " block in config_general.yml; keeping previous/default PID cut values."
                << std::endl;
    }
    return;
  }

  if (nodeParticle["CC1"])     fCC1cut    = nodeParticle["CC1"].as<double>();
  if (nodeParticle["CC2"])     fCC2cut    = nodeParticle["CC2"].as<double>();
  if (nodeParticle["PS_INIT"]) fPSInitCut = nodeParticle["PS_INIT"].as<double>();
  if (nodeParticle["PS_FIN"])  fPSFinCut  = nodeParticle["PS_FIN"].as<double>();
}

void TBaux::SetRange(const YAML::Node tConfigNode) {

  // PID integration windows (PS, MC, TC, CC1, CC2), restored from TB2025.
  // Read straight from ModuleConfig so there is a single source of truth
  // shared with `--type single --module PS` etc. Missing entries simply
  // leave that channel out of fRangeMap, and GetPIDValue()/Fill() skip it.
  for (const std::string& name : {"PS", "MC", "TC", "CC1", "CC2"}) {
    if (tConfigNode[name]) {
      const auto r = tConfigNode[name].as<std::vector<int>>();
      if (r.size() == 2) fRangeMap[name] = r;
    }
  }

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

double TBaux::GetPeakADC(std::vector<short> waveform, int xInit, int xFin, int pedBins) {
  if (pedBins <= 0 || static_cast<size_t>(pedBins) >= waveform.size())
    pedBins = 100;

  double ped = 0;
  for (int i = 1; i <= pedBins; i++)
    ped += (double)waveform.at(i) / (double)pedBins;

  std::vector<double> pedCorWave;
  for (int i = xInit; i < xFin; i++)
    pedCorWave.push_back(ped - (double)waveform.at(i));

  return *std::max_element(pedCorWave.begin(), pedCorWave.end());
}

double TBaux::GetIntADC(std::vector<short> waveform, int xInit, int xFin, int pedBins) {
  if (pedBins <= 0 || static_cast<size_t>(pedBins) >= waveform.size())
    pedBins = 100;

  double ped = 0;
  for (int i = 1; i <= pedBins; i++)
    ped += (double)waveform.at(i) / (double)pedBins;

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

std::vector<float> TBaux::GetDWCPosition(const std::vector<std::vector<float>>& wave) {

  // wave order: DWC1R, DWC1L, DWC1U, DWC1D, DWC2R, DWC2L, DWC2U, DWC2D
  if (wave.size() < 8)
    return {};

  auto binToTime = [](float bin) {
    return 800.f * (bin / 1000.f);
  };

  std::vector<float> times;
  times.reserve(8);
  for (const auto& w : wave) {
    const float bin = GetLeadingEdgeBin(w, static_cast<float>(fDWCThreshold));
    if (bin < 0) return {};
    times.push_back(binToTime(bin));
  }

  // Ported from TB2025 (see also DQM/function.h::getDWC1position /
  // getDWC2position, which carry the same constants for the legacy
  // standalone draw_*.cc tools).
  const float dwc1X = (times.at(0) - times.at(1)) * static_cast<float>(fDWCCalib.at(0)) + static_cast<float>(fDWCCalib.at(1));
  const float dwc1Y = (times.at(2) - times.at(3)) * static_cast<float>(fDWCCalib.at(2)) + static_cast<float>(fDWCCalib.at(3));
  const float dwc2X = (times.at(4) - times.at(5)) * static_cast<float>(fDWCCalib.at(4)) + static_cast<float>(fDWCCalib.at(5));
  const float dwc2Y = (times.at(6) - times.at(7)) * static_cast<float>(fDWCCalib.at(6)) + static_cast<float>(fDWCCalib.at(7));

  return {
    dwc1X - static_cast<float>(fDWCCenter.at(0)),
    dwc1Y - static_cast<float>(fDWCCenter.at(1)),
    dwc2X - static_cast<float>(fDWCCenter.at(2)),
    dwc2Y - static_cast<float>(fDWCCenter.at(3)),
  };
}

double TBaux::GetPIDValue(TBevt<TBwaveform> anEvent, const TBcid& cid, const std::string& name) {
  const auto it = fRangeMap.find(name);
  if (it == fRangeMap.end() || it->second.size() != 2) return -999.;
  const std::vector<short> wf = anEvent.GetData(cid).waveform();
  return GetValue(wf, it->second.at(0), it->second.at(1), name);
}

void TBaux::Fill(TBevt<TBwaveform> anEvent) {

  // ── Wire chamber position ───────────────────────────────────────────────
  // Skip when the loaded mapping has no WC channels (fWCEnabled == false);
  // GetData on an invalid CID would throw out_of_range.
  if (fWCEnabled && fWCPosition) {
    // Pedestal window per channel comes from PedestalBins config; if no
    // rule matches WCX/WCY/NIM the default (100) is used.
    const int pedBinsWCX = PedBinsForChannel("WCX");
    const int pedBinsWCY = PedBinsForChannel("WCY");
    const int pedBinsNIM = PedBinsForChannel("NIM");
    std::vector<std::vector<float>> wcWaves;
    wcWaves.reserve(3);
    wcWaves.push_back(anEvent.GetData(fCID_WCX).pedcorrectedWaveform(pedBinsWCX));
    wcWaves.push_back(anEvent.GetData(fCID_WCY).pedcorrectedWaveform(pedBinsWCY));
    wcWaves.push_back(anEvent.GetData(fCID_NIM).pedcorrectedWaveform(pedBinsNIM));

    const auto posVec = GetPosition(wcWaves);
    if (posVec.size() == 2)
      fWCPosition->Fill(posVec.at(0), posVec.at(1));
  }

  // ── DWC position ─────────────────────────────────────────────────────────
  if (fDWCEnabled && fDWC1Pos) {
    std::vector<std::vector<float>> dwcWaves;
    dwcWaves.reserve(8);
    dwcWaves.push_back(anEvent.GetData(fCID_DWC1R).pedcorrectedWaveform(PedBinsForChannel("DWC1R")));
    dwcWaves.push_back(anEvent.GetData(fCID_DWC1L).pedcorrectedWaveform(PedBinsForChannel("DWC1L")));
    dwcWaves.push_back(anEvent.GetData(fCID_DWC1U).pedcorrectedWaveform(PedBinsForChannel("DWC1U")));
    dwcWaves.push_back(anEvent.GetData(fCID_DWC1D).pedcorrectedWaveform(PedBinsForChannel("DWC1D")));
    dwcWaves.push_back(anEvent.GetData(fCID_DWC2R).pedcorrectedWaveform(PedBinsForChannel("DWC2R")));
    dwcWaves.push_back(anEvent.GetData(fCID_DWC2L).pedcorrectedWaveform(PedBinsForChannel("DWC2L")));
    dwcWaves.push_back(anEvent.GetData(fCID_DWC2U).pedcorrectedWaveform(PedBinsForChannel("DWC2U")));
    dwcWaves.push_back(anEvent.GetData(fCID_DWC2D).pedcorrectedWaveform(PedBinsForChannel("DWC2D")));

    const auto dwcPos = GetDWCPosition(dwcWaves);
    if (dwcPos.size() == 4) {
      fDWC1Pos->Fill(dwcPos.at(0), dwcPos.at(1));
      if (fDWC2Pos) fDWC2Pos->Fill(dwcPos.at(2), dwcPos.at(3));
      if (fDWCXCorr) fDWCXCorr->Fill(dwcPos.at(0), dwcPos.at(2));
      if (fDWCYCorr) fDWCYCorr->Fill(dwcPos.at(1), dwcPos.at(3));
    }
  }

  // ── PID (PS, MC, TC, CC1, CC2) ───────────────────────────────────────────
  if (fPIDEnabled && fPS) {
    if (fRangeMap.count("PS"))  fPS->Fill(GetPIDValue(anEvent, fCID_PS, "PS"));
    if (fRangeMap.count("MC"))  fMC->Fill(GetPIDValue(anEvent, fCID_MC, "MC"));
    if (fRangeMap.count("TC"))  fTC->Fill(GetPIDValue(anEvent, fCID_TC, "TC"));
    if (fRangeMap.count("CC1")) fCC1->Fill(GetPIDValue(anEvent, fCID_CC1, "CC1"));
    if (fRangeMap.count("CC2")) fCC2->Fill(GetPIDValue(anEvent, fCID_CC2, "CC2"));
  }

  // ── Hodoscope (16x16 IntADC and PeakADC maxima) ─────────────────────────
  if (fHodoEnabled)
    FillHodoscope(anEvent);

  // ── CERN square hodoscope (29x29, 1 mm resolution) ──────────────────────
  if (fSqHodoEnabled)
    FillSquareHodoscope(anEvent);

  // ── CERN round hodoscope (2-layer, 0.5 mm resolution) ───────────────────
  if (fRndHodoEnabled)
    FillRoundHodoscope(anEvent);
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
    // Channel names are HX1..HX16 / HY1..HY16; feed them through the
    // TBpedConfig lookup so PedestalBins.ByName/ByPrefix can override.
    const std::string nameX = "HX" + std::to_string(i + 1);
    const std::string nameY = "HY" + std::to_string(i + 1);
    const int pedBinsX = PedBinsForChannel(nameX);
    const int pedBinsY = PedBinsForChannel(nameY);
    if (wfX.size() > static_cast<size_t>(fHodoLastX[i])) {
      intADC_X[i]  = static_cast<float>(GetIntADC (wfX, fHodoFirstX[i], fHodoLastX[i], pedBinsX));
      peakADC_X[i] = static_cast<float>(GetPeakADC(wfX, fHodoFirstX[i], fHodoLastX[i], pedBinsX));
    }
    if (wfY.size() > static_cast<size_t>(fHodoLastY[i])) {
      intADC_Y[i]  = static_cast<float>(GetIntADC (wfY, fHodoFirstY[i], fHodoLastY[i], pedBinsY));
      peakADC_Y[i] = static_cast<float>(GetPeakADC(wfY, fHodoFirstY[i], fHodoLastY[i], pedBinsY));
    }
  }

  // Per-channel normalization: divide raw values by calibration constants
  // so all channels' distributions are centered near 1.0 and the
  // brightest-fiber search picks geometry, not gain.
  for (int i = 0; i < 16; ++i) {
    intADC_X[i]  /= static_cast<float>(fHodoNormIntADC_X[i]);
    intADC_Y[i]  /= static_cast<float>(fHodoNormIntADC_Y[i]);
    peakADC_X[i] /= static_cast<float>(fHodoNormPeakADC_X[i]);
    peakADC_Y[i] /= static_cast<float>(fHodoNormPeakADC_Y[i]);
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

void TBaux::GetSquareHodoRawPosition(TBevt<TBwaveform> anEvent,
                                      bool& foundInt, float& xInt, float& yInt,
                                      bool& foundPeak, float& xPeak, float& yPeak) {
  foundInt = false;
  foundPeak = false;
  if (!fSqHodoEnabled) return;
  if (fCID_SqHodoX.size() != 29 || fCID_SqHodoY.size() != 29) return;

  int bestXIdxInt = -1, bestYIdxInt = -1, bestXIdxPeak = -1, bestYIdxPeak = -1;
  double bestXValInt = -1e300, bestYValInt = -1e300, bestXValPeak = -1e300, bestYValPeak = -1e300;

  for (int i = 0; i < 29; ++i) {
    const std::string nameX = "SHX" + std::to_string(i + 1);
    const std::string nameY = "SHY" + std::to_string(i + 1);
    const int pedBinsX = PedBinsForChannel(nameX);
    const int pedBinsY = PedBinsForChannel(nameY);

    const std::vector<short> wfX = anEvent.GetData(fCID_SqHodoX[i]).waveform();
    const std::vector<short> wfY = anEvent.GetData(fCID_SqHodoY[i]).waveform();
    if (wfX.size() <= static_cast<size_t>(fSqHodoLast) || wfY.size() <= static_cast<size_t>(fSqHodoLast)) continue;

    const double rawXInt  = GetIntADC (wfX, fSqHodoFirst, fSqHodoLast, pedBinsX);
    const double rawYInt  = GetIntADC (wfY, fSqHodoFirst, fSqHodoLast, pedBinsY);
    const double rawXPeak = GetPeakADC(wfX, fSqHodoFirst, fSqHodoLast, pedBinsX);
    const double rawYPeak = GetPeakADC(wfY, fSqHodoFirst, fSqHodoLast, pedBinsY);

    // Above-threshold max search: a fiber only enters the running max if
    // its RAW ADC clears its own per-fiber threshold ("over pedestal" hit
    // requirement). Normalization is applied only to survivors.
    if (rawXInt > fSqHodoThrIntADC_X[i]) {
      const double c = rawXInt / fSqHodoNormIntADC_X[i];
      if (c > bestXValInt) { bestXValInt = c; bestXIdxInt = i; }
    }
    if (rawYInt > fSqHodoThrIntADC_Y[i]) {
      const double c = rawYInt / fSqHodoNormIntADC_Y[i];
      if (c > bestYValInt) { bestYValInt = c; bestYIdxInt = i; }
    }
    if (rawXPeak > fSqHodoThrPeakADC_X[i]) {
      const double c = rawXPeak / fSqHodoNormPeakADC_X[i];
      if (c > bestXValPeak) { bestXValPeak = c; bestXIdxPeak = i; }
    }
    if (rawYPeak > fSqHodoThrPeakADC_Y[i]) {
      const double c = rawYPeak / fSqHodoNormPeakADC_Y[i];
      if (c > bestYValPeak) { bestYValPeak = c; bestYIdxPeak = i; }
    }
  }

  // Only report a position if BOTH X and Y had a fiber above threshold;
  // otherwise the event is skipped for that metric (low hodoscope
  // efficiency means a fraction of events will have no real hit).
  if (bestXIdxInt >= 0 && bestYIdxInt >= 0) {
    foundInt = true;
    xInt = bestXIdxInt + 0.5f;
    yInt = bestYIdxInt + 0.5f;
  }
  if (bestXIdxPeak >= 0 && bestYIdxPeak >= 0) {
    foundPeak = true;
    xPeak = bestXIdxPeak + 0.5f;
    yPeak = bestYIdxPeak + 0.5f;
  }
}

void TBaux::FillSquareHodoscope(TBevt<TBwaveform> anEvent) {
  if (!fSqHodoIntADC || !fSqHodoPeakADC) return;

  bool foundInt = false, foundPeak = false;
  float xInt = 0, yInt = 0, xPeak = 0, yPeak = 0;
  GetSquareHodoRawPosition(anEvent, foundInt, xInt, yInt, foundPeak, xPeak, yPeak);

  if (foundInt)  fSqHodoIntADC ->Fill(xInt  - fSqHodoCenter[0], yInt  - fSqHodoCenter[1], 1);
  if (foundPeak) fSqHodoPeakADC->Fill(xPeak - fSqHodoCenter[0], yPeak - fSqHodoCenter[1], 1);
}

void TBaux::GetRoundHodoRawPosition(TBevt<TBwaveform> anEvent,
                                     bool& foundInt, float& xLowInt, float& yLowInt,
                                     bool& foundPeak, float& xLowPeak, float& yLowPeak) {
  foundInt = false;
  foundPeak = false;
  if (!fRndHodoEnabled) return;
  if (fCID_RndHodoX.size() != 32 || fCID_RndHodoY.size() != 32) return;

  int bestXIdxInt = -1, bestYIdxInt = -1, bestXIdxPeak = -1, bestYIdxPeak = -1;
  double bestXValInt = -1e300, bestYValInt = -1e300, bestXValPeak = -1e300, bestYValPeak = -1e300;

  for (int i = 0; i < 32; ++i) {
    // idx 0..15 -> layer 1 fiber (i+1); idx 16..31 -> layer 2 fiber (i-15).
    const int layer = (i < 16) ? 1 : 2;
    const int fiber1 = (i % 16) + 1;
    const std::string nameX = "RH" + std::to_string(layer) + "X" + std::to_string(fiber1);
    const std::string nameY = "RH" + std::to_string(layer) + "Y" + std::to_string(fiber1);
    const int pedBinsX = PedBinsForChannel(nameX);
    const int pedBinsY = PedBinsForChannel(nameY);

    const std::vector<short> wfX = anEvent.GetData(fCID_RndHodoX[i]).waveform();
    const std::vector<short> wfY = anEvent.GetData(fCID_RndHodoY[i]).waveform();
    if (wfX.size() <= static_cast<size_t>(fRndHodoLast) || wfY.size() <= static_cast<size_t>(fRndHodoLast)) continue;

    const double rawXInt  = GetIntADC (wfX, fRndHodoFirst, fRndHodoLast, pedBinsX);
    const double rawYInt  = GetIntADC (wfY, fRndHodoFirst, fRndHodoLast, pedBinsY);
    const double rawXPeak = GetPeakADC(wfX, fRndHodoFirst, fRndHodoLast, pedBinsX);
    const double rawYPeak = GetPeakADC(wfY, fRndHodoFirst, fRndHodoLast, pedBinsY);

    // Max search spans BOTH layers together: if a layer-1 and a layer-2
    // fiber both fire, the higher-ADC one wins (per-fiber threshold still
    // gates entry into the running max).
    if (rawXInt > fRndHodoThrIntADC_X[i]) {
      const double c = rawXInt / fRndHodoNormIntADC_X[i];
      if (c > bestXValInt) { bestXValInt = c; bestXIdxInt = i; }
    }
    if (rawYInt > fRndHodoThrIntADC_Y[i]) {
      const double c = rawYInt / fRndHodoNormIntADC_Y[i];
      if (c > bestYValInt) { bestYValInt = c; bestYIdxInt = i; }
    }
    if (rawXPeak > fRndHodoThrPeakADC_X[i]) {
      const double c = rawXPeak / fRndHodoNormPeakADC_X[i];
      if (c > bestXValPeak) { bestXValPeak = c; bestXIdxPeak = i; }
    }
    if (rawYPeak > fRndHodoThrPeakADC_Y[i]) {
      const double c = rawYPeak / fRndHodoNormPeakADC_Y[i];
      if (c > bestYValPeak) { bestYValPeak = c; bestYIdxPeak = i; }
    }
  }

  // Lower edge (mm) of a fiber's 1 mm footprint: layer-1 fiber n covers
  // [n-1, n], layer-2 fiber n covers [n-0.5, n+0.5] (idx encodes both
  // layer and fiber — see the (layer, fiber1) decoding above).
  auto footprintLow = [](int idx) -> float {
    const int layer = (idx < 16) ? 1 : 2;
    const int idx0 = idx % 16;
    return static_cast<float>(idx0) + (layer == 2 ? 0.5f : 0.0f);
  };

  if (bestXIdxInt >= 0 && bestYIdxInt >= 0) {
    foundInt = true;
    xLowInt = footprintLow(bestXIdxInt);
    yLowInt = footprintLow(bestYIdxInt);
  }
  if (bestXIdxPeak >= 0 && bestYIdxPeak >= 0) {
    foundPeak = true;
    xLowPeak = footprintLow(bestXIdxPeak);
    yLowPeak = footprintLow(bestYIdxPeak);
  }
}

void TBaux::FillRoundHodoscope(TBevt<TBwaveform> anEvent) {
  if (!fRndHodoIntADC || !fRndHodoPeakADC) return;

  bool foundInt = false, foundPeak = false;
  float xLowInt = 0, yLowInt = 0, xLowPeak = 0, yLowPeak = 0;
  GetRoundHodoRawPosition(anEvent, foundInt, xLowInt, yLowInt, foundPeak, xLowPeak, yLowPeak);

  // The winning fiber's 1 mm footprint always spans exactly two 0.5 mm
  // bins on each axis, so a 2D hit fills the 2x2 block of bins (4 Fill
  // calls, weight 1 each): e.g. X footprint [2.5, 3.5] -> bins centered
  // at 2.75 and 3.25.
  if (foundInt) {
    const double x = xLowInt - fRndHodoCenter[0];
    const double y = yLowInt - fRndHodoCenter[1];
    fRndHodoIntADC->Fill(x + 0.25, y + 0.25, 1);
    fRndHodoIntADC->Fill(x + 0.25, y + 0.75, 1);
    fRndHodoIntADC->Fill(x + 0.75, y + 0.25, 1);
    fRndHodoIntADC->Fill(x + 0.75, y + 0.75, 1);
  }
  if (foundPeak) {
    const double x = xLowPeak - fRndHodoCenter[0];
    const double y = yLowPeak - fRndHodoCenter[1];
    fRndHodoPeakADC->Fill(x + 0.25, y + 0.25, 1);
    fRndHodoPeakADC->Fill(x + 0.25, y + 0.75, 1);
    fRndHodoPeakADC->Fill(x + 0.75, y + 0.25, 1);
    fRndHodoPeakADC->Fill(x + 0.75, y + 0.75, 1);
  }
}

bool TBaux::PassPIDCuts(TBevt<TBwaveform> anEvent) {
  const double mcPeak  = GetPIDValue(anEvent, fCID_MC,  "MC");
  const double psPeak  = GetPIDValue(anEvent, fCID_PS,  "PS");
  const double cc1Peak = GetPIDValue(anEvent, fCID_CC1, "CC1");
  const double cc2Peak = GetPIDValue(anEvent, fCID_CC2, "CC2");

  if (mcPeak < fMCVetoThreshold) return false;

  if (fParticle == "PION" || fParticle == "KAON") {
    if (!(psPeak >= fPSInitCut && psPeak <= fPSFinCut)) return false;
    if (fCC1cut >= 0 && !(cc1Peak > fCC1cut)) return false;
    if (fCC2cut >= 0 && !(cc2Peak < fCC2cut)) return false;
  } else if (fParticle == "PROTON") {
    if (!(psPeak >= fPSInitCut && psPeak <= fPSFinCut)) return false;
    if (fCC2cut >= 0 && !(cc2Peak > fCC2cut)) return false;
  } else {
    // No --particle given: fall back to the plain PS threshold.
    if (!(psPeak >= fPSThreshold)) return false;
  }
  return true;
}

bool TBaux::IsPassing(TBevt<TBwaveform> anEvent) {

  // ── PID-only cut mode (no DWC / position requirement at all) ────────────
  // This is the fallback for years/runs where DWC is not available: same
  // MC veto + PS/CC1/CC2 particle-ID logic as the PID half of "DWCPID",
  // just without requiring a DWC position/correlation cut.
  if (fAuxCutMode == "PID") {
    if (!fPIDEnabled) return true; // degrade gracefully, like fWCEnabled above
    return PassPIDCuts(anEvent);
  }

  // ── DWC / DWCPID cut modes (restored from TB2025) ───────────────────────
  // These are independent of the WC branch below — important for CERN,
  // where there is no wire chamber at all. If DWC (or PID, for DWCPID)
  // channels are missing from the mapping, the cut is a no-op (events
  // pass through); the warning was already printed once at init time.
  if (fAuxCutMode == "DWC" || fAuxCutMode == "DWCPID") {

    if (!fDWCEnabled) return true;

    std::vector<std::vector<float>> dwcWaves;
    dwcWaves.reserve(8);
    dwcWaves.push_back(anEvent.GetData(fCID_DWC1R).pedcorrectedWaveform(PedBinsForChannel("DWC1R")));
    dwcWaves.push_back(anEvent.GetData(fCID_DWC1L).pedcorrectedWaveform(PedBinsForChannel("DWC1L")));
    dwcWaves.push_back(anEvent.GetData(fCID_DWC1U).pedcorrectedWaveform(PedBinsForChannel("DWC1U")));
    dwcWaves.push_back(anEvent.GetData(fCID_DWC1D).pedcorrectedWaveform(PedBinsForChannel("DWC1D")));
    dwcWaves.push_back(anEvent.GetData(fCID_DWC2R).pedcorrectedWaveform(PedBinsForChannel("DWC2R")));
    dwcWaves.push_back(anEvent.GetData(fCID_DWC2L).pedcorrectedWaveform(PedBinsForChannel("DWC2L")));
    dwcWaves.push_back(anEvent.GetData(fCID_DWC2U).pedcorrectedWaveform(PedBinsForChannel("DWC2U")));
    dwcWaves.push_back(anEvent.GetData(fCID_DWC2D).pedcorrectedWaveform(PedBinsForChannel("DWC2D")));

    const auto dwcPos = GetDWCPosition(dwcWaves); // {DWC1X, DWC1Y, DWC2X, DWC2Y}
    if (dwcPos.size() != 4)
      return false;

    // (1) DWC1<->DWC2 correlation cut: reject events whose trajectory
    // through DWC1 and DWC2 is inconsistent (scattered / multi-particle).
    if (fDWCCorr > 0) {
      if (std::abs(dwcPos.at(0) - dwcPos.at(2)) > fDWCCorr) return false;
      if (std::abs(dwcPos.at(1) - dwcPos.at(3)) > fDWCCorr) return false;
    }

    // (2) Optional absolute beam-spot cut on DWC1 position.
    if (fDWCPosCut > 0) {
      if (std::abs(dwcPos.at(0)) > fDWCPosCut) return false;
      if (std::abs(dwcPos.at(1)) > fDWCPosCut) return false;
    }

    // (3) DWCPID: additionally require the MC veto + PS/CC1/CC2 PID cuts.
    if (fAuxCutMode == "DWCPID") {
      if (!fPIDEnabled) return true; // degrade gracefully, like fWCEnabled above
      if (!PassPIDCuts(anEvent)) return false;
    }

    return true;
  }

  // ── WC / WCHodo cut modes (legacy, unchanged) ───────────────────────────
  // If the loaded mapping has no WC, we cannot compute the beam-spot or the
  // WC↔Hodo inclination, so --AUXcut is a no-op (let every event through).
  // The warning was printed once at init time.
  if (!fWCEnabled) return true;

  // ── (1) WC beam-spot cut (center-corrected; in mm, centered on 0) ──
  // Same pedestal-bin lookup as in Fill().
  const int pedBinsWCX = PedBinsForChannel("WCX");
  const int pedBinsWCY = PedBinsForChannel("WCY");
  const int pedBinsNIM = PedBinsForChannel("NIM");
  std::vector<std::vector<float>> wcWaves;
  wcWaves.reserve(3);
  wcWaves.push_back(anEvent.GetData(fCID_WCX).pedcorrectedWaveform(pedBinsWCX));
  wcWaves.push_back(anEvent.GetData(fCID_WCY).pedcorrectedWaveform(pedBinsWCY));
  wcWaves.push_back(anEvent.GetData(fCID_NIM).pedcorrectedWaveform(pedBinsNIM));

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

  // Only draws the WC canvas — Hodo drawing lives in Update(). If WC
  // wasn't allocated (e.g. --AUXMode Hodo), this is a no-op.
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

  // Either subsystem may have been skipped at init() time (--AUXMode WC
  // skips Hodo objects; --AUXMode Hodo skips WC objects). Return only
  // when nothing is active at all; otherwise draw what we have.
  const bool hasWC      = fWCEnabled && fCanvas && fWCPosition;
  const bool hasHodo    = fHodoEnabled && fCanvasHodoIntADC && fHodoIntADC;
  const bool hasSqHodo  = fSqHodoEnabled && fCanvasSqHodoIntADC && fCanvasSqHodoPeakADC && fSqHodoIntADC;
  const bool hasRndHodo = fRndHodoEnabled && fCanvasRndHodoIntADC && fCanvasRndHodoPeakADC && fRndHodoIntADC;
  const bool hasDWC     = fDWCEnabled && fCanvasDWC && fDWC1Pos;
  const bool hasPID     = fPIDEnabled && fCanvasPID && fPS;
  if (!hasWC && !hasHodo && !hasSqHodo && !hasRndHodo && !hasDWC && !hasPID)
    return;

  if (fIsFirst) fIsFirst = false;

  SetMaximum();

  // ── Draw all AUX canvases ───────────────────────────────────────────────
  if (hasWC) {
    fCanvas->cd(1);
    fWCPosition->Draw("colz");
    fCanvas->cd();
    fCanvas->Update();
    if (fDraw) fCanvas->Pad()->Draw();
  }

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

  if (hasSqHodo) {
    fCanvasSqHodoIntADC->cd();
    fSqHodoIntADC->Draw("colz");
    fCanvasSqHodoIntADC->Update();
    if (fDraw) fCanvasSqHodoIntADC->Pad()->Draw();

    fCanvasSqHodoPeakADC->cd();
    fSqHodoPeakADC->Draw("colz");
    fCanvasSqHodoPeakADC->Update();
    if (fDraw) fCanvasSqHodoPeakADC->Pad()->Draw();
  }

  if (hasRndHodo) {
    fCanvasRndHodoIntADC->cd();
    fRndHodoIntADC->Draw("colz");
    fCanvasRndHodoIntADC->Update();
    if (fDraw) fCanvasRndHodoIntADC->Pad()->Draw();

    fCanvasRndHodoPeakADC->cd();
    fRndHodoPeakADC->Draw("colz");
    fCanvasRndHodoPeakADC->Update();
    if (fDraw) fCanvasRndHodoPeakADC->Pad()->Draw();
  }

  if (hasDWC) {
    fCanvasDWC->cd(1); fDWC1Pos->Draw("colz");
    fCanvasDWC->cd(2); fDWC2Pos->Draw("colz");
    fCanvasDWC->cd(3); fDWCXCorr->Draw("colz");
    fCanvasDWC->cd(4); fDWCYCorr->Draw("colz");
    fCanvasDWC->cd();
    fCanvasDWC->Update();
    if (fDraw) fCanvasDWC->Pad()->Draw();
  }

  if (hasPID) {
    fCanvasPID->cd(1); fPS->Draw();
    fCanvasPID->cd(2); fMC->Draw();
    fCanvasPID->cd(3); fTC->Draw();
    fCanvasPID->cd(4);
    fCC1->Draw();
    fCC2->Draw("sames");
    TLegend* legPID = new TLegend(0.65, 0.7, 0.9, 0.9);
    legPID->SetFillStyle(0);
    legPID->SetBorderSize(0);
    legPID->AddEntry(fCC1, "CC1", "l");
    legPID->AddEntry(fCC2, "CC2", "l");
    legPID->Draw();
    fCanvasPID->cd();
    fCanvasPID->Update();
    if (fDraw) fCanvasPID->Pad()->Draw();
  }

  // ── Combined AUX ROOT file (WC + hodoscope + DWC + PID) ─────────────────
  // Only write the subsystems that were actually allocated. E.g. a WC-only
  // run produces a file containing only fCanvas_WC + WC_Position; a
  // DWC-only run produces only the DWC canvas/histograms.
  TString output = "./output/Run" + std::to_string(fRunNum) + "_AUX.root";
  if (fAuxCut) output = "./output/Run" + std::to_string(fRunNum) + "_AUX_AuxCut.root";
  {
    TFile outoutFile(output, "RECREATE");
    outoutFile.cd();
    if (hasWC) {
      fCanvas->Write();
      fWCPosition->Write();
    }
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
    if (hasSqHodo) {
      fCanvasSqHodoIntADC->Write();
      fCanvasSqHodoPeakADC->Write();
      fSqHodoIntADC->Write();
      fSqHodoPeakADC->Write();
    }
    if (hasRndHodo) {
      fCanvasRndHodoIntADC->Write();
      fCanvasRndHodoPeakADC->Write();
      fRndHodoIntADC->Write();
      fRndHodoPeakADC->Write();
    }
    if (hasDWC) {
      fCanvasDWC->Write();
      fDWC1Pos->Write();
      fDWC2Pos->Write();
      fDWCXCorr->Write();
      fDWCYCorr->Write();
    }
    if (hasPID) {
      fCanvasPID->Write();
      fPS->Write();
      fMC->Write();
      fTC->Write();
      fCC1->Write();
      fCC2->Write();
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
  //   method=Hodoscope  → fCanvas_SqHodoIntADC      (CERN square 29x29, IntADC)
  //   method=Hodoscope  → fCanvas_SqHodoPeakADC     (CERN square 29x29, PeakADC)
  //   method=Hodoscope  → fCanvas_RndHodoIntADC     (CERN round 33x33, IntADC)
  //   method=Hodoscope  → fCanvas_RndHodoPeakADC    (CERN round 33x33, PeakADC)
  //   method=DWC        → fCanvas_DWC               (DWC1/2 position + correlation)
  //   method=PID        → fCanvas_PID               (PS/MC/TC/CC1/CC2)
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
  if (hasSqHodo) {
    dumpJSON(fCanvasSqHodoIntADC,  "Hodoscope");
    dumpJSON(fCanvasSqHodoPeakADC, "Hodoscope");
  }
  if (hasRndHodo) {
    dumpJSON(fCanvasRndHodoIntADC,  "Hodoscope");
    dumpJSON(fCanvasRndHodoPeakADC, "Hodoscope");
  }
  if (hasDWC) dumpJSON(fCanvasDWC, "DWC");
  if (hasPID) dumpJSON(fCanvasPID, "PID");

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

  // Nothing to save if no subsystem produced an object.
  if (!fWCPosition && !fHodoIntADC && !fSqHodoIntADC && !fRndHodoIntADC && !fDWC1Pos && !fPS)
    return;

  TFile* outoutFile = new TFile(output, "RECREATE");
  outoutFile->cd();

  if (fWCPosition) fWCPosition->Write();
  if (fHodoEnabled) {
    if (fHodoIntADC)       fHodoIntADC      ->Write();
    if (fHodoPeakADC)      fHodoPeakADC     ->Write();
    if (fHodoIntADC_corr)  fHodoIntADC_corr ->Write();
    if (fHodoPeakADC_corr) fHodoPeakADC_corr->Write();
  }
  if (fSqHodoEnabled) {
    if (fSqHodoIntADC)  fSqHodoIntADC ->Write();
    if (fSqHodoPeakADC) fSqHodoPeakADC->Write();
  }
  if (fRndHodoEnabled) {
    if (fRndHodoIntADC)  fRndHodoIntADC ->Write();
    if (fRndHodoPeakADC) fRndHodoPeakADC->Write();
  }
  if (fDWCEnabled) {
    if (fDWC1Pos)  fDWC1Pos ->Write();
    if (fDWC2Pos)  fDWC2Pos ->Write();
    if (fDWCXCorr) fDWCXCorr->Write();
    if (fDWCYCorr) fDWCYCorr->Write();
  }
  if (fPIDEnabled) {
    if (fPS)  fPS ->Write();
    if (fMC)  fMC ->Write();
    if (fTC)  fTC ->Write();
    if (fCC1) fCC1->Write();
    if (fCC2) fCC2->Write();
  }

  outoutFile->Close();
}
