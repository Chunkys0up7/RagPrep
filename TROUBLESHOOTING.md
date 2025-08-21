# 🐛 Troubleshooting Guide - RAG Document Processing Utility

This guide helps you resolve common issues when setting up and running the RAG Document Processing Utility.

## 🚨 **Common Issues & Solutions**

### **1. Dependency Installation Failures**

#### **Problem**: `pip install -r requirements.txt` fails
```
❌ Failed to install dependencies: Command returned non-zero exit status 1
```

#### **Solutions**:

**A. Try Minimal Dependencies First**
```bash
# Install only essential packages
pip install -r requirements_minimal.txt

# Then try the full requirements
pip install -r requirements.txt
```

**B. Upgrade pip and setuptools**
```bash
python -m pip install --upgrade pip setuptools wheel
```

**C. Install with Verbose Output**
```bash
pip install -r requirements.txt -v
```
This shows exactly which package is failing and why.

**D. Use Alternative Package Index**
```bash
pip install -r requirements.txt -i https://pypi.org/simple/
```

**E. Install System Dependencies (Linux/macOS)**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev build-essential

# macOS
xcode-select --install
```

### **2. Python Version Issues**

#### **Problem**: Python version compatibility
```
❌ Python 3.8 or higher is required
```

#### **Solutions**:

**A. Check Your Python Version**
```bash
python --version
python3 --version
```

**B. Use Python 3.8-3.11 (Recommended)**
- Python 3.13 may have compatibility issues with some packages
- Python 3.8-3.11 are fully tested and supported

**C. Create Virtual Environment with Specific Python Version**
```bash
# If you have multiple Python versions
python3.11 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### **3. Package-Specific Issues**

#### **PyTorch Installation Problems**
```bash
# Try CPU-only version first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or use conda
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

#### **Transformers Installation Problems**
```bash
# Install with specific version
pip install transformers==4.35.2

# Or skip heavy dependencies for basic functionality
pip install transformers --no-deps
```

#### **OpenCV Installation Problems**
```bash
# Try alternative package
pip install opencv-python-headless

# Or skip for basic functionality
```

### **4. Network and SSL Issues**

#### **Problem**: SSL certificate errors or network timeouts
```
SSL: CERTIFICATE_VERIFY_FAILED
```

#### **Solutions**:

**A. Trusted Hosts**
```bash
pip install -r requirements.txt --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org
```

**B. Use HTTP (if HTTPS fails)**
```bash
pip install -r requirements.txt -i http://pypi.org/simple/
```

**C. Configure pip.conf**
```ini
# ~/.pip/pip.conf or %APPDATA%\pip\pip.ini
[global]
trusted-host = pypi.org pypi.python.org files.pythonhosted.org
```

### **5. Platform-Specific Issues**

#### **Windows Issues**

**A. Microsoft Visual C++ Required**
```
error: Microsoft Visual C++ 14.0 or greater is required
```
**Solution**: Install [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

**B. Path Length Issues**
```
error: [Errno 2] No such file or directory
```
**Solution**: Use shorter paths or enable long path support in Windows

**C. PowerShell Execution Policy**
```
File cannot be loaded because running scripts is disabled
```
**Solution**: 
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### **macOS Issues**

**A. Xcode Command Line Tools**
```
clang: error: no input files
```
**Solution**: 
```bash
xcode-select --install
```

**B. Homebrew Dependencies**
```bash
brew install openssl readline sqlite3 xz zlib
```

#### **Linux Issues**

**A. System Libraries**
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev build-essential libssl-dev libffi-dev

# CentOS/RHEL
sudo yum install python3-devel gcc openssl-devel libffi-devel
```

### **6. Alternative Installation Methods**

#### **Use Conda Instead of pip**
```bash
# Install Miniconda first, then:
conda create -n ragprep python=3.11
conda activate ragprep
conda install -c conda-forge pydantic pyyaml pathlib2
```

#### **Use Poetry for Dependency Management**
```bash
# Install Poetry first, then:
poetry install
```

#### **Install Packages Individually**
```bash
# Core packages
pip install pydantic pydantic-settings pyyaml pathlib2

# ML packages (optional for basic functionality)
pip install torch transformers sentence-transformers

# Other packages
pip install fastapi uvicorn requests
```

### **7. Simplified Demo for Troubleshooting**

If you're still having issues, try the simplified demo:

```bash
# Run with minimal dependencies
python demo_simple.py
```

This demo:
- ✅ Works with minimal packages
- ✅ Tests core functionality
- ✅ Creates sample output
- ✅ Identifies specific issues

### **8. Getting Help**

#### **Check the Logs**
- Look for specific error messages in the terminal output
- Check `demo.log` for detailed error information
- Review the verbose pip output (`pip install -v`)

#### **Common Error Patterns**
```
ERROR: Could not find a version that satisfies the requirement
→ Package version compatibility issue

ERROR: Microsoft Visual C++ 14.0 or greater is required
→ Windows build tools missing

ERROR: Failed building wheel for [package]
→ System dependencies missing

ERROR: [SSL: CERTIFICATE_VERIFY_FAILED]
→ Network/SSL configuration issue
```

#### **Still Stuck?**
1. **Check the GitHub Issues** for similar problems
2. **Try the simplified demo** to isolate the issue
3. **Share the complete error message** for specific help
4. **Check your Python and pip versions** for compatibility

## 🎯 **Quick Fix Checklist**

- [ ] **Python Version**: 3.8-3.11 (avoid 3.13)
- [ ] **pip**: Latest version (`python -m pip install --upgrade pip`)
- [ ] **Virtual Environment**: Use isolated environment
- [ ] **System Dependencies**: Install build tools for your platform
- [ ] **Minimal Dependencies**: Try `requirements_minimal.txt` first
- [ ] **Network**: Check firewall/proxy settings
- [ ] **Alternative Methods**: Try conda or individual package installation

## 🎉 **Success Indicators**

When everything is working correctly, you should see:
```
✅ All core dependencies are available
✅ Project structure is correct
✅ Test documents are readable
✅ Core modules imported successfully
✅ Basic functionality test passed
🎉 Simplified demo completed successfully!
```

---

**💡 Pro Tip**: Start with the simplified demo (`demo_simple.py`) to verify basic functionality, then gradually add more dependencies as needed.
