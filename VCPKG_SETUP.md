# vcpkg Setup Instructions

This project uses **vcpkg** for dependency management.

## Prerequisites

- Visual Studio 2019 or later (with C++ workload)
- Git
- PowerShell (comes with Windows)

## One-Time Setup

### Step 1: Install vcpkg

Choose a location for vcpkg (e.g., `C:\dev\vcpkg` or `C:\tools\vcpkg`):

```powershell
# Navigate to where you want vcpkg installed
cd C:\dev

# Clone vcpkg repository
git clone https://github.com/Microsoft/vcpkg.git

# Navigate into vcpkg directory
cd vcpkg

# Bootstrap vcpkg (builds vcpkg executable)
.\bootstrap-vcpkg.bat
```

### Step 2: Integrate vcpkg with Visual Studio

```powershell
# From the vcpkg directory
.\vcpkg integrate install
```

This command:
- Sets up vcpkg integration with Visual Studio/MSBuild
- Enables automatic dependency resolution for projects with `vcpkg.json`
- Only needs to be run once per machine

**Note:** If you see a message about user-wide integration, that's normal. The integration applies to all Visual Studio projects on your machine.

### Step 3: Install Project Dependencies

From the **project root directory** (where `vcpkg.json` is located):

```powershell
# Make sure you're in the project root
cd C:\Users\me\src\opencl-windows

# Install dependencies (vcpkg will detect vcpkg.json automatically)
vcpkg install
```

Or if vcpkg is not in your PATH:

```powershell
# Replace with your actual vcpkg path
C:\dev\vcpkg\vcpkg.exe install
```

This will:
- Read `vcpkg.json` from the current directory
- Download and install OpenCL headers and libraries
- Create a `vcpkg_installed/` directory (already in `.gitignore`)

## Building the Project

After setup, you can build normally in Visual Studio:

1. Open `main.sln` in Visual Studio
2. Build the project (F7 or Build → Build Solution)
3. vcpkg dependencies will be automatically resolved

## Troubleshooting

### "vcpkg: command not found"

Add vcpkg to your PATH, or use the full path:
```powershell
$env:PATH += ";C:\dev\vcpkg"
```

Or create an alias in your PowerShell profile:
```powershell
function vcpkg { & "C:\dev\vcpkg\vcpkg.exe" $args }
```

### "Cannot find OpenCL headers"

1. Make sure `vcpkg integrate install` was run
2. Verify `vcpkg.json` exists in project root
3. Run `vcpkg install` from project root
4. Restart Visual Studio

### "Cannot find OpenCL.lib"

vcpkg should automatically provide the library path. If issues persist:
1. Check that `VcpkgEnableManifest` is set to `true` in `.vcxproj`
2. Verify vcpkg integration: `vcpkg integrate show`
3. Rebuild the solution

### Clean Build

If you need to start fresh:
```powershell
# Remove vcpkg installed packages for this project
Remove-Item -Recurse -Force vcpkg_installed

# Reinstall
vcpkg install
```

## Updating Dependencies

To update OpenCL to a newer version, edit `vcpkg.json`:

```json
{
  "dependencies": [
    {
      "name": "opencl",
      "version>=": "2024.01.01"  // Update version here
    }
  ]
}
```

Then run:
```powershell
vcpkg install
```

## Additional Resources

- [vcpkg Documentation](https://github.com/Microsoft/vcpkg)
- [vcpkg Getting Started](https://github.com/Microsoft/vcpkg/blob/master/README.md)
- [vcpkg Manifest Mode](https://learn.microsoft.com/en-us/vcpkg/consume/manifest-mode)
