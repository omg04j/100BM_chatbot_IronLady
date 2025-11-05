"""
Advanced Vector Store for Iron Lady Leadership Program (100BM)
===============================================================
Optimized for your folder structure with intelligent metadata extraction

Features:
- Session-aware chunking
- Community content recognition
- Facilitator extraction
- Smart metadata tagging
- Hierarchical organization
- OpenAI embeddings for best RAG performance

Author: Optimized for 100BM RAG System
Date: November 2025
"""

import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document



# ============================================================================
# METADATA EXTRACTORS
# ============================================================================

class MetadataExtractor:
    """Extract rich metadata from file names and paths"""
    
    def __init__(self):
        # YouTube URL patterns
        self.youtube_patterns = [
            re.compile(r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})', re.IGNORECASE),
            re.compile(r'(?:youtube|utube|video|link|url)\s*(?:url)?\s*[:=-]\s*(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})', re.IGNORECASE),
        ]
    
    def extract_youtube_urls(self, text: str) -> List[str]:
        """Extract YouTube URLs from text"""
        video_ids = set()
        for pattern in self.youtube_patterns:
            matches = pattern.findall(text)
            video_ids.update(matches)
        return [f"https://www.youtube.com/watch?v={vid}" for vid in video_ids]
    
    @staticmethod
    def extract_session_number(file_path: str) -> Optional[int]:
        """Extract session number from path or filename"""
        patterns = [
            r'Session\s*(\d+)',
            r'session\s*(\d+)',
            r'Session-(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, file_path, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None
    
    @staticmethod
    def extract_facilitator(filename: str) -> Optional[str]:
        """Extract facilitator name from filename"""
        # Patterns like "by Name" or "- Name"
        patterns = [
            r'by\s+([A-Z][a-zA-Z\s]+?)(?:\.|$)',
            r'-\s*([A-Z][a-zA-Z\s]+?)\.docx',
            r'Showcase\s*-\s*([A-Z][a-zA-Z\s]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                name = match.group(1).strip()
                # Clean up common words
                if name not in ['Session', 'Image', 'Iron', 'Lady', 'Way']:
                    return name
        
        return None
    
    @staticmethod
    def get_content_type(file_path: str) -> str:
        """Determine content type based on path"""
        path_lower = file_path.lower()
        
        if 'session 1' in path_lower and 'image creation' in path_lower:
            return 'session_1_image_creation'
        elif 'session 2' in path_lower and '4t' in path_lower:
            return 'session_2_4t_management'
        elif 'session 3' in path_lower and 'capability' in path_lower:
            return 'session_3_capability'
        elif 'session 4' in path_lower and 'pitch' in path_lower:
            return 'session_4_pitch'
        elif 'session 5' in path_lower and 'bing fa' in path_lower:
            return 'session_5_strategy'
        elif 'session 6' in path_lower:
            return 'session_6_implementation'
        elif '100 bm community' in path_lower:
            return 'community_session'
        elif 'lep revision' in path_lower:
            return 'revision_session'
        else:
            return 'general'
    
    @staticmethod
    def get_session_title(file_path: str, session_num: Optional[int]) -> str:
        """Get descriptive session title"""
        titles = {
            1: "Image Creation - Iron Lady Way",
            2: "4T Management for Operational Excellence",
            3: "Breakthrough Capability Development",
            4: "Pitch Without Pitching & Influencing Tactics",
            5: "Bing Fa Stratagem (11-Point Framework)",
            6: "Real-Time Implementation & Board Tips"
        }
        
        if session_num and session_num in titles:
            return titles[session_num]
        
        # Extract from filename
        filename = Path(file_path).name
        if '-' in filename:
            parts = filename.split('-', 1)
            if len(parts) > 1:
                return parts[1].replace('.docx', '').strip()
        
        return filename.replace('.docx', '').strip()
    
    @staticmethod
    def is_boardroom_showcase(filename: str) -> bool:
        """Check if file is a boardroom showcase"""
        return 'boardroom' in filename.lower() and 'showcase' in filename.lower()
    
    @staticmethod
    def is_boardroom_sawaal(filename: str) -> bool:
        """Check if file is boardroom Q&A"""
        return 'boardroom sawaal' in filename.lower()
    
    @staticmethod
    def is_industry_leader_connect(filename: str) -> bool:
        """Check if file is industry leader session"""
        return 'industry leader connect' in filename.lower()


# ============================================================================
# ADVANCED DOCUMENT PROCESSOR
# ============================================================================

class DocumentProcessor:
    """Process documents with intelligent chunking and metadata"""
    
    def __init__(self):
        self.metadata_extractor = MetadataExtractor()
        
        # Text splitter - optimized for course content with SECTION awareness
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,  # Larger chunks for structured content
            chunk_overlap=300,  # More overlap to preserve context
            length_function=len,
            separators=[
                "\n\n## SECTION",  # Section breaks
                "\n\n### ",         # Sub-section breaks
                "\n\n",             # Paragraph breaks
                "\n",               # Line breaks
                ". ",               # Sentences
                " ",                # Words
                ""                  # Characters
            ]
        )
    
    def extract_section_info(self, text: str, chunk: str) -> Dict[str, Optional[str]]:
        """
        Extract section information from structured content
        Looks for patterns like:
        - SECTION 1: TITLE
        - ## SECTION 3: TARGET MANAGEMENT
        - ### Subsection Title
        """
        section_info = {
            'section_number': None,
            'section_title': None,
            'subsection': None
        }
        
        # Find which section this chunk belongs to
        # Look backwards from chunk position in full text
        chunk_pos = text.find(chunk[:50])  # Find approximate position
        
        if chunk_pos != -1:
            # Look for last section header before this chunk
            text_before = text[:chunk_pos]
            
            # Pattern: SECTION 1: TITLE or ## SECTION 1: TITLE
            section_pattern = r'(?:##\s*)?SECTION\s+(\d+):\s*([^\n]+)'
            section_matches = list(re.finditer(section_pattern, text_before, re.IGNORECASE))
            
            if section_matches:
                last_section = section_matches[-1]
                section_info['section_number'] = int(last_section.group(1))
                section_info['section_title'] = last_section.group(2).strip()
            
            # Look for subsection (###)
            subsection_pattern = r'###\s+([^\n]+)'
            subsection_matches = list(re.finditer(subsection_pattern, text_before))
            
            if subsection_matches:
                last_subsection = subsection_matches[-1]
                section_info['subsection'] = last_subsection.group(1).strip()
        
        return section_info
    
    def load_document(self, file_path: Path) -> List[Document]:
        """Load a single document with rich metadata"""
        try:
            # Load document
            loader = Docx2txtLoader(str(file_path))
            raw_docs = loader.load()
            
            if not raw_docs:
                return []
            
            full_content = raw_docs[0].page_content
            
            if not full_content.strip():
                return []
            
            # Extract metadata
            session_num = self.metadata_extractor.extract_session_number(str(file_path))
            facilitator = self.metadata_extractor.extract_facilitator(file_path.name)
            content_type = self.metadata_extractor.get_content_type(str(file_path))
            session_title = self.metadata_extractor.get_session_title(str(file_path), session_num)
            
            # Determine category
            parent_folder = file_path.parent.name
            
            # Split into chunks
            chunks = self.text_splitter.split_text(full_content)
            
            processed_docs = []
            
            for idx, chunk in enumerate(chunks):
                # Extract section info from structured content
                section_info = self.extract_section_info(full_content, chunk)
                
                # Enhanced metadata for each doc
                metadata = {
                    'source_file': file_path.name,
                    'file_path': str(file_path),
                    'parent_folder': parent_folder,
                    'session_number': session_num,
                    'session_title': session_title,
                    'facilitator': facilitator,
                    'content_type': content_type,
                    'is_boardroom_showcase': self.metadata_extractor.is_boardroom_showcase(file_path.name),
                    'is_boardroom_sawaal': self.metadata_extractor.is_boardroom_sawaal(file_path.name),
                    'is_industry_leader': self.metadata_extractor.is_industry_leader_connect(file_path.name),
                    'processed_date': datetime.now().isoformat(),
                    'chunk_index': idx,
                    # NEW: Section-aware metadata
                    'document_section_number': section_info['section_number'],
                    'document_section_title': section_info['section_title'],
                    'document_subsection': section_info['subsection'],
                }
                
                # Add category tags
                if parent_folder == '100 BM Community':
                    metadata['category'] = 'community'
                elif parent_folder == 'LEP revision':
                    metadata['category'] = 'revision'
                elif session_num:
                    metadata['category'] = f'session_{session_num}'
                else:
                    metadata['category'] = 'general'
                
                # Enhanced content with section context
                content_prefix = []
                
                if session_num:
                    content_prefix.append(f"Session {session_num}: {session_title}")
                
                if facilitator:
                    content_prefix.append(f"Facilitator: {facilitator}")
                
                if section_info['section_title']:
                    content_prefix.append(f"Section {section_info['section_number']}: {section_info['section_title']}")
                
                if section_info['subsection']:
                    content_prefix.append(f"Topic: {section_info['subsection']}")
                
                content_prefix.append(f"File: {file_path.name}")
                
                enhanced_content = "\n".join(content_prefix) + "\n\n" + chunk
                
                doc = Document(
                    page_content=enhanced_content,
                    metadata=metadata
                )
                processed_docs.append(doc)
            
            return processed_docs
            
        except Exception as e:
            print(f"  ⚠️  Error loading {file_path.name}: {e}")
            return []
    
    def create_chunks(self, documents: List[Document]) -> List[Document]:
        """Split documents into optimized chunks"""
        return self.text_splitter.split_documents(documents)


# ============================================================================
# MAIN VECTOR STORE CREATOR
# ============================================================================

class VectorStoreCreator:
    """Create optimized vector store from structured content"""
    
    def __init__(
        self,
        content_folder: str = "./lms_content",
        vector_store_path: str = "./vector_store",
        embedding_model: str = "openai"
    ):
        self.content_folder = Path(content_folder)
        self.vector_store_path = vector_store_path
        
        # Setup embeddings
        if embedding_model == "openai":
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=st.secrets.get("OPENAI_API_KEY")
            )
        else:
            raise ValueError("Only OpenAI embeddings supported for best performance")
        
        # Document processor
        self.processor = DocumentProcessor()
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'total_documents': 0,
            'total_chunks': 0,
            'by_category': {},
            'by_session': {},
            'facilitators': set()
        }
    
    def discover_files(self) -> List[Path]:
        """Discover all .docx files in content folder"""
        print(f"\n📂 Discovering files in: {self.content_folder}")
        
        if not self.content_folder.exists():
            print(f"❌ Error: Folder not found!")
            return []
        
        # Find all .docx files recursively
        files = list(self.content_folder.glob("**/*.docx"))
        
        # Filter out temp files
        files = [f for f in files if not f.name.startswith('~')]
        
        print(f"✓ Found {len(files)} .docx files")
        
        # Show structure
        folders = {}
        for file in files:
            folder = file.parent.name
            folders[folder] = folders.get(folder, 0) + 1
        
        print("\n📊 File distribution:")
        for folder, count in sorted(folders.items()):
            print(f"   • {folder}: {count} files")
        
        return files
    
    def load_all_documents(self, files: List[Path]) -> List[Document]:
        """Load all documents with metadata"""
        print(f"\n📖 Loading documents...")
        
        all_documents = []
        
        for i, file_path in enumerate(files, 1):
            print(f"   [{i}/{len(files)}] {file_path.parent.name}/{file_path.name}")
            
            docs = self.processor.load_document(file_path)
            
            if docs:
                all_documents.extend(docs)
                
                # Update stats
                self.stats['total_files'] += 1
                
                # Track by category
                category = docs[0].metadata.get('category', 'unknown')
                self.stats['by_category'][category] = self.stats['by_category'].get(category, 0) + 1
                
                # Track by session
                session = docs[0].metadata.get('session_number')
                if session:
                    self.stats['by_session'][session] = self.stats['by_session'].get(session, 0) + 1
                
                # Track facilitators
                facilitator = docs[0].metadata.get('facilitator')
                if facilitator:
                    self.stats['facilitators'].add(facilitator)
        
        self.stats['total_documents'] = len(all_documents)
        print(f"\n✓ Loaded {len(all_documents)} documents from {self.stats['total_files']} files")
        
        return all_documents
    
    def create_chunks(self, documents: List[Document]) -> List[Document]:
        """Create optimized chunks"""
        print(f"\n✂️  Creating chunks...")
        
        chunks = self.processor.create_chunks(documents)
        self.stats['total_chunks'] = len(chunks)
        
        print(f"✓ Created {len(chunks)} chunks")
        
        return chunks
    
    def create_vector_store(self, chunks: List[Document]) -> Chroma:
        """Create and persist vector store"""
        print(f"\n🔧 Creating vector store...")
        print(f"   Location: {self.vector_store_path}")
        print(f"   Chunks: {len(chunks)}")
        print(f"   Embedding: OpenAI")
        
        # Create vector store
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.vector_store_path
        )
        
        # Verify
        count = vector_store._collection.count()
        print(f"✓ Vector store created with {count} vectors")
        
        return vector_store
    
    def print_statistics(self):
        """Print comprehensive statistics"""
        print("\n" + "="*80)
        print("📊 VECTOR STORE STATISTICS")
        print("="*80)
        
        print(f"\n📁 Files & Documents:")
        print(f"   • Total Files: {self.stats['total_files']}")
        print(f"   • Total Documents: {self.stats['total_documents']}")
        print(f"   • Total Chunks: {self.stats['total_chunks']}")
        
        print(f"\n📚 By Category:")
        for category, count in sorted(self.stats['by_category'].items()):
            print(f"   • {category}: {count} files")
        
        if self.stats['by_session']:
            print(f"\n🎓 By Session:")
            for session in sorted(self.stats['by_session'].keys()):
                count = self.stats['by_session'][session]
                print(f"   • Session {session}: {count} files")
        
        if self.stats['facilitators']:
            print(f"\n👥 Facilitators Found:")
            for facilitator in sorted(self.stats['facilitators']):
                print(f"   • {facilitator}")
        
        print("\n" + "="*80)
    
    def run(self):
        """Execute complete pipeline"""
        print("="*80)
        print("🚀 Iron Lady Leadership Program - Vector Store Creator")
        print("="*80)
        print("\n✨ Features:")
        print("   • Session-aware metadata")
        print("   • Facilitator extraction")
        print("   • Category tagging")
        print("   • Optimized chunking")
        print("   • OpenAI embeddings (best for RAG)")
        print("="*80)
        
        # Pipeline
        files = self.discover_files()
        if not files:
            print("\n❌ No files found. Exiting.")
            return
        
        documents = self.load_all_documents(files)
        if not documents:
            print("\n❌ No documents loaded. Exiting.")
            return
        
        chunks = self.create_chunks(documents)
        if not chunks:
            print("\n❌ No chunks created. Exiting.")
            return
        
        self.create_vector_store(chunks)
        
        # Show statistics
        self.print_statistics()
        
        print("\n✅ SUCCESS! Vector store created and ready to use!")
        print("\n📝 Next Steps:")
        print("   1. Run: python utils.py")
        print("   2. Ask questions about the Iron Lady Leadership Program")
        print("   3. RAG will use session context, facilitators, and categories")
        print("="*80)


# ============================================================================
# VECTOR STORE LOADER (for utils.py)
# ============================================================================

class VectorStoreLoader:
    """Load existing vector store for querying"""
    
    def __init__(self, vector_store_path: str = "./vector_store"):
        self.vector_store_path = vector_store_path
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=st.secrets.get("OPENAI_API_KEY")
        )
        self.vector_store = None
    
    def load(self) -> Chroma:
        """Load existing vector store"""
        print(f"📂 Loading vector store from: {self.vector_store_path}")
        
        self.vector_store = Chroma(
            persist_directory=self.vector_store_path,
            embedding_function=self.embeddings
        )
        
        count = self.vector_store._collection.count()
        print(f"✓ Loaded {count} vectors")
        
        return self.vector_store
    
    def search(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Document]:
        """Search vector store"""
        if not self.vector_store:
            self.load()
        
        if filter_dict:
            return self.vector_store.similarity_search(query, k=k, filter=filter_dict)
        else:
            return self.vector_store.similarity_search(query, k=k)
    
    def search_by_session(self, query: str, session_number: int, k: int = 5) -> List[Document]:
        """Search within specific session"""
        return self.search(query, k=k, filter_dict={'session_number': session_number})
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        if not self.vector_store:
            self.load()
        
        all_docs = self.vector_store.get()
        
        stats = {
            'total_vectors': len(all_docs['ids']),
            'sessions': set(),
            'categories': set(),
            'facilitators': set()
        }
        
        for metadata in all_docs['metadatas']:
            if metadata.get('session_number'):
                stats['sessions'].add(metadata['session_number'])
            if metadata.get('category'):
                stats['categories'].add(metadata['category'])
            if metadata.get('facilitator'):
                stats['facilitators'].add(metadata['facilitator'])
        
        return stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    
    # Create vector store
    creator = VectorStoreCreator(
        content_folder="./lms_content",
        vector_store_path="./vector_store",
        embedding_model="openai"
    )
    
    creator.run()


if __name__ == "__main__":

    main()


