#ifndef TBaux_h
#define TBaux_h 1

#include <map>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <chrono>
#include <cmath>
#include <numeric>
#include <functional>

#include "TBconfig.h"
#include "TBpedConfig.h"
#include "TButility.h"
#include "TBdetector.h"
#include "TBplotengine.h"
#include "TBmid.h"
#include "TBevt.h"

#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TLegend.h"

class TBaux
{
public:
  TBaux() = default;
  TBaux(const YAML::Node fNodePlot_, int fRunNum_, bool fPlotting_, bool fLive_, bool fDraw_, TButility fUtility_);
  ~TBaux() {}

  void init();

  void Fill(TBevt<TBwaveform> anEvent);
  void Fill(TBevt<TBfastmode> anEvent) {}

  float LinearInterp(float x1, float y1, float x2, float y2, float threshold) const;
  float GetLeadingEdgeBin(const std::vector<float>& waveform, float percent) const;
  std::vector<float> GetPosition(const std::vector<std::vector<float>>& wave); // WCX, WCY, NIM

  // DWC (Delayed Wire Chamber) position reconstruction, restored from
  // TB2025. `wave` must contain 8 pedestal-corrected waveforms in the
  // order [DWC1R, DWC1L, DWC1U, DWC1D, DWC2R, DWC2L, DWC2U, DWC2D].
  // Returns {DWC1X, DWC1Y, DWC2X, DWC2Y} in mm (center-corrected via
  // AUX.DWC.CENTER), or an empty vector if any leading edge is not found.
  std::vector<float> GetDWCPosition(const std::vector<std::vector<float>>& wave);



  void Draw();
  void Update();

  void SaveAs(TString output = "");

  // Hodoscope filling (16x16 fiber map of the highest-amplitude X/Y bin per
  // event). Called from Fill() when --AUX is set.
  void FillHodoscope(TBevt<TBwaveform> anEvent);

  // Brightest-fiber positions in fiber coordinates (= mm; 1 fiber = 1 mm).
  // Returns { x_intADC, y_intADC, x_peakADC, y_peakADC } as raw bin
  // centers (range 0..16). Empty vector if the hodoscope is disabled or
  // channel data is missing. Used both by FillHodoscope() and by
  // IsPassing() (for the WC+Hodo inclination cut).
  std::vector<float> GetHodoscopeRawPosition(TBevt<TBwaveform> anEvent);

  // CERN square hodoscope (29 X-fibers x 29 Y-fibers, 1 mm resolution,
  // SHX1..29 / SHY1..29). Called from Fill() when --AUXMode includes
  // "Hodo" and the square-hodo channels resolved from the mapping.
  void FillSquareHodoscope(TBevt<TBwaveform> anEvent);

  // CERN round hodoscope (2 staggered layers x 16 fibers per axis, 0.5 mm
  // resolution, RH1X/RH1Y/RH2X/RH2Y 1..16). Called from Fill() under the
  // same conditions as FillSquareHodoscope().
  void FillRoundHodoscope(TBevt<TBwaveform> anEvent);

  // Pedestal window length used by Get{Int,Peak}ADC. Configurable per
  // channel name through config_general.yml::PedestalBins (see TBpedConfig).
  // Default 100 bins if no rule matches.
  double GetPeakADC(std::vector<short> waveform, int xInit, int xFin, int pedBins = 100);
  double GetIntADC(std::vector<short> waveform, int xInit, int xFin, int pedBins = 100);

  int PedBinsForChannel(const std::string& name) const { return fPedConfig.BinsFor(name); }

  double GetValue(std::vector<short> waveform, int xInit, int xFin, const std::string& name = "") {

    const int pedBins = PedBinsForChannel(name);

    if(fMethod == "PeakADC")
      return GetPeakADC(waveform, xInit, xFin, pedBins);

    if(fMethod == "IntADC")
      return GetIntADC(waveform, xInit, xFin, pedBins);

    return -999;
  }

  // See TBplotengine::SetPedestalBins for semantics.
  void SetPedestalBins(const YAML::Node& node) { fPedConfig.Load(node); }

  std::vector<int> GetUniqueMID() {

    return fUtility.GetUniqueMID(fCIDtoPlot);
  }

  void SetRange(const YAML::Node tConfigNode);
  void SetMethod(std::string fMethod_) { 
    fMethod = fMethod_; 
    if (fMethod == "Overlay" || fMethod == "Avg") fMethod = "IntADC";
  }
  void SetApp(TApplication* fApp_) { fApp = fApp_; }
  void SetAUXCut(bool fAuxCut_) { fAuxCut = fAuxCut_; }
  // AUXcut mode: "WC" applies only the WC POSCUT (legacy behavior; WC is
  // currently hidden from the web UI since CERN has no wire chamber, but
  // the code path is kept for sites that do). "WCHodo" additionally
  // requires |WC_corr − Hodo_corr| < INCLINATION_CUT on both axes. "DWC"
  // applies the DWC1<->DWC2 correlation cut (+ optional DWC1 absolute
  // POSCUT), independent of WC. "PID" applies only the MC veto + PS/CC1/CC2
  // particle-ID cuts (see SetParticle()), with no position requirement at
  // all — the fallback for runs/years where DWC is not available. "DWCPID"
  // is "DWC" + "PID" combined (both the position and particle-ID cuts).
  // Any unrecognised value falls back to "WC".
  void SetAUXCutMode(const std::string& fAuxCutMode_) { fAuxCutMode = fAuxCutMode_; }
  // AUX scope mode: a comma-separated list of any of "WC", "Hodo", "DWC",
  // "PID" (e.g. "DWC,PID", "WC,DWC"), or the legacy single tokens "WC" |
  // "Hodo" | "WCHodo" (kept for backward compatibility; "WCHodo" expands
  // to "WC,Hodo"). Selects which AUX subsystems to plot. Must be called
  // BEFORE init() so the CID resolution can skip subsystems the operator
  // didn't ask for — critical when a subsystem (e.g. hodoscope, or WC/DWC
  // at a site where they're not installed) is physically absent from the
  // setup. Tokens that don't resolve to a known subsystem are ignored with
  // a warning; if nothing valid is found the whole thing falls back to
  // "WC,Hodo" inside init().
  void SetAUXMode(const std::string& fAuxMode_) { fAuxMode = fAuxMode_; }
  void SetParticle(std::string fParticle_);

  bool IsPassing(TBevt<TBwaveform> anEvent);

  void SetMaximum();

private:
  // Looks up ModuleConfig[name] as the integration window and evaluates
  // GetValue() (fMethod, always PeakADC for AUX — see TBmonit) on the raw
  // waveform of `cid`. Returns -999 if `name` has no configured window.
  // Used for the PS/MC/TC/CC1/CC2 PID plots and the DWCPID event cut.
  double GetPIDValue(TBevt<TBwaveform> anEvent, const TBcid& cid, const std::string& name);

  // MC veto + PS/CC1/CC2 particle-ID cuts, shared by the "PID" and
  // "DWCPID" --AUXCutMode branches (the only difference between the two
  // is whether a DWC position/correlation cut is also required). Assumes
  // fPIDEnabled == true; callers must check that first.
  bool PassPIDCuts(TBevt<TBwaveform> anEvent);

  // Per-event brightest-above-threshold fiber search for the CERN square
  // hodoscope, done independently for IntADC and PeakADC. A fiber only
  // enters the max search if its RAW (pre-normalization) ADC exceeds its
  // configured per-fiber threshold ("over pedestal" hit requirement); if
  // no X and/or no Y fiber passes for a given metric, found{Int,Peak} is
  // left false for that metric (the event is skipped — low hodoscope
  // efficiency is expected). x/y are raw fiber-center coordinates (mm,
  // range 0..29).
  void GetSquareHodoRawPosition(TBevt<TBwaveform> anEvent,
                                 bool& foundInt, float& xInt, float& yInt,
                                 bool& foundPeak, float& xPeak, float& yPeak);

  // Same idea for the CERN round hodoscope, searching across BOTH layers
  // together per axis. x/y are the LOWER edge (mm, range 0..16.5) of the
  // winning fiber's 1 mm footprint (not its center) — FillRoundHodoscope()
  // uses this to fill the 2x2 block of 0.5 mm bins the footprint covers.
  void GetRoundHodoRawPosition(TBevt<TBwaveform> anEvent,
                                bool& foundInt, float& xLowInt, float& yLowInt,
                                bool& foundPeak, float& xLowPeak, float& yLowPeak);

  const YAML::Node fNodeAux;
  int fRunNum;
  bool fPlotting;
  bool fLive;
  bool fDraw;
  bool fAuxCut;
  // AUXcut mode: "WC" (default) or "WCHodo". See SetAUXCutMode().
  std::string fAuxCutMode;
  // AUX scope mode: "WC" | "Hodo" | "WCHodo". See SetAUXMode().
  // Default "WCHodo" preserves legacy behaviour: --AUX without an
  // explicit --AUXMode still plots both subsystems.
  std::string fAuxMode;
  // Beam-inclination cut (mm) for the WCHodo mode. Read from
  // AUX.INCLINATION_CUT; defaults to [4, 4]. Stored as 2 entries
  // [X_cut, Y_cut]; if the YAML provides fewer entries the default
  // is kept.
  std::vector<double> fInclinationCut;
  std::string fParticle;

  TButility fUtility;

  TApplication* fApp;
  TCanvas* fCanvas;

  bool fIsFirst;

  std::string fMethod;

  TH2D* fWCPosition;

  double fWCThreshold;
  double fWCCalibration;
  std::vector<double> fWCReference; // timing reference per axis
  double fWCPosCut;

  std::vector<TBcid> fCIDtoPlot;
  std::map<std::string, std::vector<int>> fRangeMap;

  // True only when all three WC channels (WCX/WCY/NIM) resolved to valid
  // CIDs in the loaded mapping. Used to gate Fill()/IsPassing() so a mapping
  // without WC (e.g. mapping_TB2025_v1.root for MCPPMT runs) doesn't crash
  // when --AUX or --AUXcut is requested.
  bool fWCEnabled;
  TBcid fCID_WCX;
  TBcid fCID_WCY;
  TBcid fCID_NIM;

  // ── DWC (Delayed Wire Chamber): 2 chambers x 4 timing edges each ────────
  // Restored from TB2025. Loaded when --AUXMode includes "DWC" and/or
  // --AUXCutMode is "DWC"/"DWCPID". Degrades gracefully (fDWCEnabled=false,
  // no MID read) if DWC1R/L/U/D + DWC2R/L/U/D are absent from the mapping —
  // e.g. when the DWC hardware is not installed for a given run.
  bool fDWCEnabled;
  TBcid fCID_DWC1R, fCID_DWC1L, fCID_DWC1U, fCID_DWC1D;
  TBcid fCID_DWC2R, fCID_DWC2L, fCID_DWC2U, fCID_DWC2D;
  double fDWCThreshold; // leading-edge fraction, AUX.DWC.THRESHOLD
  // [1X_slope, 1X_intercept, 1Y_slope, 1Y_intercept,
  //  2X_slope, 2X_intercept, 2Y_slope, 2Y_intercept], AUX.DWC.CALIB
  std::vector<double> fDWCCalib;
  std::vector<double> fDWCCenter; // [1X, 1Y, 2X, 2Y] offsets, AUX.DWC.CENTER
  double fDWCCorr;   // max |DWC1-DWC2| per axis (mm) for the correlation cut
  double fDWCPosCut; // optional |DWC1 X|,|DWC1 Y| absolute cut (mm); <=0 off
  TH2D* fDWC1Pos;   // DWC1 (X, Y)
  TH2D* fDWC2Pos;   // DWC2 (X, Y)
  TH2D* fDWCXCorr;  // DWC1 X vs DWC2 X
  TH2D* fDWCYCorr;  // DWC1 Y vs DWC2 Y
  TCanvas* fCanvasDWC;

  // ── PID (PS, MC, TC, CC1, CC2) ───────────────────────────────────────────
  // Restored from TB2025. Loaded when --AUXMode includes "PID" and/or
  // --AUXCutMode is "DWCPID". Degrades gracefully if PS/MC/TC/CC1/CC2 are
  // absent from the mapping.
  bool fPIDEnabled;
  TBcid fCID_PS, fCID_MC, fCID_TC, fCID_CC1, fCID_CC2;
  TH1D* fPS;
  TH1D* fMC;
  TH1D* fTC;
  TH1D* fCC1;
  TH1D* fCC2;
  TCanvas* fCanvasPID;
  double fPSThreshold;     // AUX.PID.PS_THRESHOLD_PEAKADC ("no --particle" cut)
  double fMCVetoThreshold; // AUX.PID.MC_VETO_PEAKADC
  // Particle-specific PID windows, set by SetParticle() from
  // AUX.<PARTICLE>.{CC1,CC2,PS_INIT,PS_FIN}. CC1 < 0 means "not used"
  // (e.g. PROTON selection only requires CC2).
  double fCC1cut;
  double fCC2cut;
  double fPSInitCut;
  double fPSFinCut;

  // ── Hodoscope (16 X-fibers × 16 Y-fibers) ───────────────────────────────
  // Channel-name lists currently use the dummy tower-channel names from
  // draw_hodoscope.cc; once the proper hodoscope mapping is delivered they
  // can be swapped for X1..X16 / Y1..Y16 without touching the logic.
  bool fHodoEnabled;
  std::vector<TBcid> fCID_HodoX;   // 16 entries
  std::vector<TBcid> fCID_HodoY;   // 16 entries
  // Per-fiber search-window [first, last] for the brightest-fiber scan.
  // The same window feeds both GetIntADC and GetPeakADC for that fiber.
  // Read from ModuleConfig.HX1..HX16 / HY1..HY16 in SetRange(); falls back
  // to (150, 350) per fiber when an entry is missing.
  std::vector<int> fHodoFirstX;    // 16 entries
  std::vector<int> fHodoLastX;     // 16 entries
  std::vector<int> fHodoFirstY;    // 16 entries
  std::vector<int> fHodoLastY;     // 16 entries
  // Reference fiber position (X_ref, Y_ref) where the beam-center sits
  // before any correction. Read from AUX.Hodoscope.CENTER; defaults to
  // the nominal center (8, 8) which means "no correction".
  std::vector<float> fHodoCenter;
  // Which brightest-fiber metric feeds the WC↔Hodo inclination cut.
  // Read from AUX.Hodoscope.CUT_METHOD; "IntADC" (default) or "PeakADC".
  // Unrelated to fMethod (which controls the *main* DQM plots).
  std::string fHodoCutMethod;
  // Per-channel normalization constants read from config_general.yml
  // (AUX.Hodoscope.NORM_CONST_INTADC / NORM_CONST_PEAKADC).
  // calibrated = raw / norm_const; all 1.0 means no normalization.
  std::vector<double> fHodoNormIntADC_X;   // 16 entries (HX1..HX16)
  std::vector<double> fHodoNormIntADC_Y;   // 16 entries (HY1..HY16)
  std::vector<double> fHodoNormPeakADC_X;  // 16 entries (HX1..HX16)
  std::vector<double> fHodoNormPeakADC_Y;  // 16 entries (HY1..HY16)
  // Raw hit-map of the brightest X/Y fiber per event.
  TH2F* fHodoIntADC;
  TH2F* fHodoPeakADC;
  TCanvas* fCanvasHodoIntADC;
  TCanvas* fCanvasHodoPeakADC;
  // Same maps, shifted by -fHodoCenter + (8, 8) so the beam appears at
  // the nominal hodoscope center.
  TH2F* fHodoIntADC_corr;
  TH2F* fHodoPeakADC_corr;
  TCanvas* fCanvasHodoIntADC_corr;
  TCanvas* fCanvasHodoPeakADC_corr;

  // ── CERN Square Hodoscope (29 X-fibers x 29 Y-fibers, 1 mm resolution) ──
  // SHX1..29 / SHY1..29, active area 29x29 mm^2. Loaded under the same
  // needHodo gate as the legacy 16x16 hodoscope above (both are controlled
  // by the single "Hodo" AUXMode token / web checkbox).
  bool fSqHodoEnabled;
  std::vector<TBcid> fCID_SqHodoX;   // 29 entries (SHX1..SHX29)
  std::vector<TBcid> fCID_SqHodoY;   // 29 entries (SHY1..SHY29)
  // Shared integration window for all 58 channels. Read from
  // AUX.Hodoscope.SQUARE.RANGE; defaults to (150, 350).
  int fSqHodoFirst;
  int fSqHodoLast;
  // Reference point (X_ref, Y_ref), in raw fiber-center mm (range 0..29),
  // subtracted from the raw position before filling. Read from
  // AUX.Hodoscope.SQUARE.CENTER; default (14.5, 14.5) = active-area
  // center, i.e. no shift (histograms are drawn centered on 0).
  std::vector<double> fSqHodoCenter;
  // Per-channel gain-normalization constants (calibrated = raw / norm).
  // Read from AUX.Hodoscope.SQUARE.NORM_CONST_{INTADC,PEAKADC}.{SHX,SHY}.
  std::vector<double> fSqHodoNormIntADC_X;   // 29 entries
  std::vector<double> fSqHodoNormIntADC_Y;   // 29 entries
  std::vector<double> fSqHodoNormPeakADC_X;  // 29 entries
  std::vector<double> fSqHodoNormPeakADC_Y;  // 29 entries
  // Per-fiber "over pedestal" hit thresholds, compared against the RAW
  // (pre-normalization) ADC. Read from
  // AUX.Hodoscope.SQUARE.THRESHOLD_{INTADC,PEAKADC}.{SHX,SHY}; default 0
  // for every channel (no rejection) until a calibration run tunes them.
  std::vector<double> fSqHodoThrIntADC_X;    // 29 entries
  std::vector<double> fSqHodoThrIntADC_Y;    // 29 entries
  std::vector<double> fSqHodoThrPeakADC_X;   // 29 entries
  std::vector<double> fSqHodoThrPeakADC_Y;   // 29 entries
  TH2F* fSqHodoIntADC;
  TH2F* fSqHodoPeakADC;
  TCanvas* fCanvasSqHodoIntADC;   // separate single-pad canvas, IntADC only
  TCanvas* fCanvasSqHodoPeakADC;  // separate single-pad canvas, PeakADC only

  // ── CERN Round Hodoscope (2 layers x 16 fibers per axis, 0.5 mm res.) ───
  // RH1X/RH1Y/RH2X/RH2Y, 1..16 each; active area 16.5x16.5 mm^2.
  bool fRndHodoEnabled;
  // 32 entries each: index 0..15 = layer-1 fibers 1..16, index 16..31 =
  // layer-2 fibers 1..16. Kept in this fixed order everywhere below.
  std::vector<TBcid> fCID_RndHodoX;
  std::vector<TBcid> fCID_RndHodoY;
  // Shared integration window for all 64 channels. Read from
  // AUX.Hodoscope.ROUND.RANGE; defaults to (150, 350).
  int fRndHodoFirst;
  int fRndHodoLast;
  // Reference point (X_ref, Y_ref), in raw mm (range 0..16.5). Read from
  // AUX.Hodoscope.ROUND.CENTER; default (8.25, 8.25) = active-area center
  // (no shift).
  std::vector<double> fRndHodoCenter;
  // Per-channel gain-normalization constants, 32 entries each (layer 1
  // then layer 2). Read from
  // AUX.Hodoscope.ROUND.NORM_CONST_{INTADC,PEAKADC}.{RH1X,RH2X,RH1Y,RH2Y}.
  std::vector<double> fRndHodoNormIntADC_X;
  std::vector<double> fRndHodoNormIntADC_Y;
  std::vector<double> fRndHodoNormPeakADC_X;
  std::vector<double> fRndHodoNormPeakADC_Y;
  // Per-fiber hit thresholds (RAW ADC), 32 entries each, default 0. Read
  // from AUX.Hodoscope.ROUND.THRESHOLD_{INTADC,PEAKADC}.{RH1X,RH2X,RH1Y,RH2Y}.
  std::vector<double> fRndHodoThrIntADC_X;
  std::vector<double> fRndHodoThrIntADC_Y;
  std::vector<double> fRndHodoThrPeakADC_X;
  std::vector<double> fRndHodoThrPeakADC_Y;
  TH2F* fRndHodoIntADC;
  TH2F* fRndHodoPeakADC;
  TCanvas* fCanvasRndHodoIntADC;   // separate single-pad canvas, IntADC only
  TCanvas* fCanvasRndHodoPeakADC;  // separate single-pad canvas, PeakADC only

  // Per-channel pedestal bin window; see TBplotengine::fPedConfig.
  TBpedConfig fPedConfig;
};








#endif
