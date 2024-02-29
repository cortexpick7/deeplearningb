import json
import os
import variables

bcolors = variables.bcolors

print(bcolors.OKBLUE + "Starting program setup..." + bcolors.ENDC)
print("\n")
print("Task: Obtaining settings from file...")
pathToSettings = "settings.json"
settingsJsonString = open(pathToSettings)
desrializedJson = json.load(settingsJsonString)
print(bcolors.OKGREEN + "Task: Success" + bcolors.ENDC)
print("\n")
print("Please write absolute path to data folder.")
print(bcolors.WARNING + "Please note that data folder must contain subfolders: \"Train\", \"Test\"." + bcolors.ENDC)
pathToData = input()
print("Task: Generate paths to subdirs...")
desrializedJson["dataFolder"] = os.path.join(pathToData)
desrializedJson["trainFolder"] = os.path.join(pathToData, "Train")
desrializedJson["testFolder"] = os.path.join(pathToData, "Test")
print(bcolors.OKGREEN + "Task: Success" + bcolors.ENDC)
print("\n")
print("Please write absolute path to folder where you want to store logs.")
pathToLogs = input()
desrializedJson["logsFolder"] = pathToLogs
print("\n")
print("Please write absolute path to folder where you want to store models.")
pathToModels = input()
desrializedJson["modelsFolder"] = pathToModels
with open(pathToSettings, 'w') as json_file:
  json.dump(desrializedJson, json_file)
print(bcolors.OKGREEN + "Setup complete." + bcolors.ENDC)


