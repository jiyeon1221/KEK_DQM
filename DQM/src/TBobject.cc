#include "TBobject.h"
#include "Rtypes.h"

ObjectCollection::ObjectCollection(int argc, char* argv[])
: fArgc(argc)
{
  for (int i = 1; i < fArgc; i++) {
    fArgv.push_back(std::string(argv[i]));
    // std::cout << "Print " << i - 1 << " " << fArgv.at(i - 1) << std::endl;
  }

  init();

  for (int i = 0; i < fArgv.size(); i++) {

    if (fArgv[i] == "--RunNumber") {
      i++;
      AddVariable("RunNumber", std::stoi(fArgv[i]));
    }

    if (fArgv[i] == "--MaxEvent") {
      i++;
      AddVariable("MaxEvent", std::stoi(fArgv[i]));
    }

    if (fArgv[i] == "--SkipEvent") {
      i++;
      AddVariable("SkipEvent", std::stoi(fArgv[i]));
    }

    if (fArgv[i] == "--MaxFile") {
      i++;
      AddVariable("MaxFile", std::stoi(fArgv[i]));
    }

    if (fArgv[i] == "--type") {
      i++;
      AddVariable("type", fArgv[i]);
    }

    if (fArgv[i] == "--particle") {
      i++;
      AddVariable("particle", fArgv[i]);
    }

    if (fArgv[i] == "--method") {
      i++;
      AddVariable("method", fArgv[i]);
    }

    if (fArgv[i] == "--module") {
      while(i < fArgv.size()) {
        i++;

        if (fArgv[i].find("--") != std::string::npos || fArgv[i] == "" || i == fArgv.size()) {
          i--;
          break;
        }

        AddVec("module", fArgv[i]);
      }
    }

    if (fArgv[i] == "--Config") {
      i++;
      AddVariable("Config", fArgv[i]);
    }

    if (fArgv[i] == "--AUXcut")
      AddVariable("AUXcut", true);

    if (fArgv[i] == "--AUXCutMode") {
      i++;
      AddVariable("AUXCutMode", fArgv[i]);
    }

    if (fArgv[i] == "--AUX")
      AddVariable("AUX", true);

    if (fArgv[i] == "--LIVE")
      AddVariable("LIVE", true);

    if (fArgv[i] == "--DRAW")
      AddVariable("DRAW", true);

  }
}

bool ObjectCollection::Help() {

  if (std::string(fArgv[0]) == "--h" || std::string(fArgv[0]) == "-h" || std::string(fArgv[0]) == "-help") {
    ANSI_CODE ANSI = ANSI_CODE();

    std::cout << ANSI.BOLD << "### How To Use" << ANSI.END << std::endl;
    std::cout << ANSI.BOLD << "------------------------------------------------------" << ANSI.END << std::endl;

    std::cout << ANSI.BOLD + ANSI.YELLOW + "  --RunNumber" + ANSI.END << " : which run want to draw" << std::endl;
    std::cout << "    eg) ./monit --RunNumber 9119"<< std::endl;
    std::cout << ANSI.BOLD << "------------------------------------------------------" << ANSI.END << std::endl;

    std::cout << ANSI.BOLD + ANSI.YELLOW + "  --type" + ANSI.END << " : what we want to draw" << std::endl;
    std::cout << ANSI.BOLD + "    allowed type: " + ANSI.YELLOW + "'single', 'heatmap'" + ANSI.END << std::endl;
    std::cout << "    eg) ./monit --RunNumver 9119 --type single"<< std::endl;
    std::cout << ANSI.BOLD << "------------------------------------------------------" << ANSI.END << std::endl;

    std::cout << ANSI.BOLD + ANSI.YELLOW + "  --method" + ANSI.END << " : which method for calculating ADC" << std::endl;
    std::cout << ANSI.BOLD + "    allowed method: " + ANSI.END << std::endl;
    std::cout << ANSI.BOLD + "      --type single: " + ANSI.YELLOW + "'IntADC', 'PeakADC', 'Avg', 'Overlay'" + ANSI.END << std::endl;
    std::cout << ANSI.BOLD + "      --type heatmap: " + ANSI.YELLOW + "'IntADC', 'PeakADC'" + ANSI.END << std::endl;
    std::cout << "    eg) ./monit --RunNumver 9119 --type single --method IntADC" << std::endl;
    std::cout << ANSI.BOLD << "------------------------------------------------------" << ANSI.END << std::endl;

    std::cout << ANSI.BOLD + ANSI.YELLOW + "  Mandatory options for '--type single'" + ANSI.END << std::endl;
    std::cout << ANSI.BOLD + "    --module " + ANSI.YELLOW + "module name ..." + ANSI.END << std::endl;
    std::cout << "    eg) ./monit --RunNumver 9119 --type single --method IntADC --module M1T1_S M1T1_C M1T2_S M1T2_C" << std::endl;
    std::cout << ANSI.BOLD << "------------------------------------------------------" << ANSI.END << std::endl;

    std::cout << ANSI.BOLD + ANSI.YELLOW + "  Mandatory options for '--method heatmap'" + ANSI.END << std::endl;
    std::cout << ANSI.BOLD + "    --module " + ANSI.YELLOW + "'SiPM', 'MCPPMT'" + ANSI.END << std::endl;
    std::cout << "    eg) ./monit --RunNumver 9119 --type heatmap --method IntADC --module SiPM" << std::endl;
    std::cout << ANSI.BOLD << "------------------------------------------------------" << ANSI.END << std::endl;

    std::cout << ANSI.BOLD + ANSI.YELLOW + "  if want to draw AUX info '--AUX'" + ANSI.END << std::endl;
    std::cout << "    eg) ./monit --RunNumver 9119 --type heatmap --method IntADC --module SiPM --AUX" << std::endl;
    std::cout << ANSI.BOLD << "------------------------------------------------------" << ANSI.END << std::endl;

    std::cout << ANSI.BOLD + ANSI.YELLOW + "  if want to do live monitoring '--LIVE'" + ANSI.END << std::endl;
    std::cout << "    eg) ./monit --RunNumver 9119 --type heatmap --method IntADC --module SiPM --LIVE" << std::endl;
    std::cout << ANSI.BOLD << "------------------------------------------------------" << ANSI.END << std::endl;

    return true;
  }

  return false;
}

void ObjectCollection::GetVariable(std::string key, bool* value) {

  if (fVarBool.find(key) == fVarBool.end()) *value = false;
  else                                      *value = fVarBool[key];
}

void ObjectCollection::GetVariable(std::string key, int* value) {

  if (fVarInt.find(key) == fVarInt.end()) *value = -1;
  else                                    *value = fVarInt[key];
}

void ObjectCollection::GetVariable(std::string key, double* value) {

  if (fVarDouble.find(key) == fVarDouble.end()) *value = -1;
  else                                          *value = fVarDouble[key];
}

void ObjectCollection::GetVariable(std::string key, std::string* value) {

  if (fVarString.find(key) == fVarString.end()) *value = "null";
  else                                          *value = fVarString[key];
}

void ObjectCollection::GetVector(std::string key, std::vector<int>* value) {
  *value = fVarIntVec[key];

  if (fVarIntVec.find(key) == fVarIntVec.end()) *value = std::vector<int>{};
  else                                          *value = fVarIntVec[key];
}

void ObjectCollection::GetVector(std::string key, std::vector<double>* value) {

  if (fVarDoubleVec.find(key) == fVarDoubleVec.end()) *value = std::vector<double>{};
  else                                                *value = fVarDoubleVec[key];
}

void ObjectCollection::GetVector(std::string key, std::vector<std::string>* value) {

  if (fVarStringVec.find(key) == fVarStringVec.end()) *value = std::vector<std::string>{};
  else                                                *value = fVarStringVec[key];
}

void ObjectCollection::init() {

  fVarBool = {};
  fVarInt = {};
  fVarIntVec = {};
  fVarDouble = {};
  fVarDoubleVec = {};
  fVarString = {};
  fVarStringVec = {};
}

void ObjectCollection::Print() {

  for (int i = 0; i < fArgv.size(); i++)
    std::cout << i << " " << fArgv.at(i) << std::endl;

  for (auto varBool : fVarBool)
    std::cout << "fVarBool: " << varBool.first << " " << varBool.second << std::endl;

  for (auto varInt : fVarInt)
    std::cout << "fVarInt: " << varInt.first << " " << varInt.second << std::endl;

  for (auto varDouble : fVarDouble)
    std::cout << "fVarDouble: " << varDouble.first << " " << varDouble.second << std::endl;

  for (auto varStr : fVarString)
    std::cout << "fVarString: " << varStr.first << " " << varStr.second << std::endl;

  for (auto varInt : fVarIntVec) {
    std::cout << "fVarIntVec: " + varInt.first;
    for (int i = 0; i < varInt.second.size(); i++)
      std::cout << " " << varInt.second.at(i) << " |";
  }
  std::cout << " " << std::endl;

  for (auto varInt : fVarDoubleVec) {
    std::cout << "fVarDoubleVec: " + varInt.first;
    for (int i = 0; i < varInt.second.size(); i++)
      std::cout << " " << varInt.second.at(i) << " |";
  }
  std::cout << " " << std::endl;

  for (auto varInt : fVarStringVec) {
    std::cout << "fVarStringVec: " + varInt.first;
    std::cout << " " + std::to_string(varInt.second.size());
    for (int i = 0; i < varInt.second.size(); i++)
      std::cout << " " << varInt.second.at(i) << " |";
  }
  std::cout << " " << std::endl;
}
