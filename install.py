"""
Installation script for Deep Researcher Agent.
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    return True

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    directories = [
        "data",
        "data/embeddings", 
        "data/documents",
        "examples",
        "tests"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def test_installation():
    """Test the installation."""
    print("Testing installation...")
    try:
        # Test imports
        from deep_researcher import DeepResearcher
        print("✓ DeepResearcher import successful")
        
        # Test basic initialization
        researcher = DeepResearcher(data_dir="test_install_data")
        print("✓ DeepResearcher initialization successful")
        
        # Clean up
        researcher.close()
        import shutil
        shutil.rmtree("test_install_data", ignore_errors=True)
        
        print("✓ Installation test passed")
        return True
    except Exception as e:
        print(f"Error testing installation: {e}")
        return False

def main():
    """Main installation function."""
    print("Deep Researcher Agent - Installation Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("Installation failed. Please check the error messages above.")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Test installation
    if not test_installation():
        print("Installation test failed. Please check the error messages above.")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("Installation completed successfully!")
    print("\nTo get started:")
    print("1. Run: python main.py")
    print("2. Or run: python examples/basic_usage.py")
    print("3. For web interface: python main.py --mode web")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
