import launch

if not launch.is_installed("Face Swapper"):
    launch.run_pip("install insightface==0.7.1", "requirements for Face Swapper")
