#ifndef TBmonit_h
#define TBmonit_h 1

#include <string>
#include <iostream>

#include "TFile.h"
#include "TBconfig.h"
#include "TBread.h"
#include "TBobject.h"

template <class T>
class TBmonit
{
public:
  TBmonit(const std::string &fConfig_, int fRunNum_);
  TBmonit(ObjectCollection* obj);

  ~TBmonit() {}

  void Loop();
  void LoopLive();
  void LoopAfterRun();
  // AstroPix-only run: --type Astro. No TBread/TBplotengine/mapping at all
  // -- just TBastro::Run() (synchronous) + Update(). Dispatched from
  // Loop() before the LIVE/after-run branch, since it doesn't need any of
  // the usual main-loop setup (run number + max event are all it needs).
  void LoopAstroOnly();

  // void LoopFast(); //FIXME!! Fast engine should be integrated into TBplotengine, and work using template!!

  void SetMaxEvent(int fMaxEvent_) { fMaxEvent = fMaxEvent_; }
  void SetMaxFile(int fMaxFile_) { fMaxFile = fMaxFile_; }
  void SetLive() { fIsLive = true; }
  void GetFormattedRamInfo();

private:
  TBconfig fConfig;

  ObjectCollection* fObj;
  TButility fUtility;

  TApplication* fApp;

  std::string fBaseDir;
  std::string fMapping;
  std::string fParticle;

  int fRunNum;
  int fMaxEvent;
  int fMaxFile;

  bool fIsLive;
  bool fDraw;
  bool fAuxPlotting;
  bool fAuxCut;
  // --Astro: draw the AstroPix 2D hitmap using TBastro/TBAstroReader, in a
  // parallel thread alongside the main loop (LoopAfterRun only -- LIVE
  // mode prints a warning and skips it, see TBastro's header comment for
  // why: TBAstroReader has no chunked/next-file-waiting support yet).
  // Independent of fAuxPlotting/fAuxCut -- AstroPix does not use the
  // channel mapping or TBaux at all.
  bool fAstro;
  // AUXcut mode: "WC" (default — WC beam-spot cut only) or "WCHodo"
  // (additionally applies the WC↔hodoscope inclination cut). Set via
  // --AUXCutMode and forwarded to TBaux::SetAUXCutMode().
  std::string fAuxCutMode;
  // AUX scope: which subsystems to plot when --AUX is on. One of
  // "WC", "Hodo", "WCHodo" (default WCHodo for legacy behavior).
  // Set via --AUXMode and forwarded to TBaux::SetAUXMode() before
  // init() so TBaux can skip loading the unused subsystem's MIDs.
  std::string fAuxMode;
};

#endif
