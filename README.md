# Email_Classification_System
An automated email classification system using SVM and GloVe embeddings integrated into Outlook using Visual Studio VSTO (Visual Studio Tools for Office) Add-in

Core Technology Stack
	•	Machine Learning: Support Vector Machine (SVM) with RBF kernel
	•	Natural Language Processing: GloVe word embeddings (100-dimensional vectors)
	•	Integration: VSTO Add-in with C# for Microsoft Outlook
	•	Development Tools: Python (ML model), Visual Studio (Outlook integration)
Key Technical Components
	•	Email preprocessing pipeline with stop word removal and word stemming
	•	Pre-trained GloVe embeddings for semantic text analysis
	•	SVM classifier with optimized parameters (C=100, gamma=0.01)
	•	Asynchronous email processing to maintain Outlook performance
	•	Exchange Online PowerShell deployment scripts
