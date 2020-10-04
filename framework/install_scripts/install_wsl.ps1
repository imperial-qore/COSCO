<#
   This script will install WSL in Windows.
   Adapted from: https://github.com/electronicegg/wsl-ubuntu-powershell
   Last updated: 2020-10-02
#>

<# Enable WSL feature in Windows #>
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux

<# Download Ubuntu #>
curl.exe -L -o ubuntu-1804.appx https://aka.ms/wsl-ubuntu-1804
Rename-Item ubuntu-1804.appx ubuntu-1804.zip

<# Install Ubuntu #>
Expand-Archive ubuntu-1804.zip ubuntu
$userenv = [System.Environment]::GetEnvironmentVariable("Path", "User")
[System.Environment]::SetEnvironmentVariable("PATH", $userenv + (get-location) + "\ubuntu", "User")
.\ubuntu\ubuntu1804.exe

<# Reset Ubuntu #>
wslconfig.exe /u Ubuntu-18.04
Remove-Item .\ubuntu\
Expand-Archive ubuntu-1804.zip ubuntu
.\ubuntu\ubuntu1804.exe
