#ifndef TBastro_h
#define TBastro_h 1

// ─────────────────────────────────────────────────────────────────────────────
//  TBastro
//
//  AstroPix 2D hitmap, run in parallel with the main TBread-based event loop
//  (see TBmonit::LoopAfterRun / LoopAstroOnly). AstroPix's ASIC-defined data
//  layout does not fit the standard TBread/TBaux mapping-based path, so this
//  class is fully independent: its own reader (TBAstroReader), no TBcid /
//  TButility, no shared TBread instance — and, critically, no ROOT calls
//  until Update(), which must run on the main thread after the worker
//  thread returned from Run(). Run() itself only decodes files and
//  accumulates a plain count array, so it is safe to execute on a
//  std::thread alongside ROOT-touching code in the main loop.
//
//  Typical usage from TBmonit:
//    TBastro astro(fConfig.GetConfig()["Astro"], fRunNum, fBaseDir);
//    std::thread t(&TBastro::Run, &astro, fMaxEvent);
//    ... (main loop runs concurrently) ...
//    t.join();
//    astro.Update();
// ─────────────────────────────────────────────────────────────────────────────

#include <array>
#include <cstdint>
#include <string>

#include "yaml-cpp/yaml.h"

class TBastro {
public:
  // fGlobalBaseDir_ is config_general.yml's top-level BaseDirectory, used as
  // a fallback when Astro.BaseDirectory is not set in the config, so a site
  // that keeps AstroPix data in the standard DAQ tree doesn't need to add
  // anything beyond the Astro: block itself.
  TBastro(const YAML::Node& fNodeAstro_, int fRunNum_, const std::string& fGlobalBaseDir_);

  // Thread entry point. maxEvent == -1 means "all available events"; the
  // Astro reader resolves its own total independently of the main loop's
  // reader (the two data sources are not required to have matching event
  // counts). Prints "Astro <i> / <n> events (<pct> %)" progress lines to
  // stdout every ~2000 events. Touches no ROOT object.
  void Run(int maxEvent);

  // Builds the TH2F/TCanvas from the accumulated counts and writes
  // ./output/Run<N>_Astro.root + ./output/Run<N>_Astro_Hitmap_<canvas>.json.
  // Must be called from the main thread, after Run() has returned (join()).
  // No-op if the Astro data directory/files were not found (Run() already
  // printed a warning in that case).
  void Update();

  bool HasData() const { return fHasData; }
  long GetNEvents() const { return fNEvents; }
  long GetNHits() const { return fNHits; }
  long GetNPairs() const { return fNPairs; }

private:
  static constexpr int kNCol = 35;
  static constexpr int kNRow = 35;

  int fRunNum;
  int fMID;
  int fTsDiff;
  double fTotDiffPct;
  std::string fBaseDir;

  bool fHasData;
  long fNEvents;
  long fNHits;
  long fNPairs;

  std::array<std::array<std::uint64_t, kNRow>, kNCol> fCounts;
};

#endif
