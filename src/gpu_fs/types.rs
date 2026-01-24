// Core types and constants for GPU-Native Filesystem

/// Block size (4KB - standard for most filesystems)
pub const BLOCK_SIZE: usize = 4096;

/// Root inode ID (always 0)
pub const ROOT_INODE_ID: u32 = 0;

/// Invalid inode sentinel value
pub const INVALID_INODE: u32 = 0xFFFFFFFF;

/// Maximum path depth (prevents infinite loops)
pub const MAX_PATH_DEPTH: usize = 16;

/// Maximum filename length (same as most filesystems)
pub const MAX_FILENAME_LEN: usize = 255;

/// File types
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FileType {
    Regular = 0,
    Directory = 1,
    Symlink = 2,
    BlockDevice = 3,
    CharDevice = 4,
    Fifo = 5,
    Socket = 6,
    Unknown = 7,
}

impl FileType {
    pub fn from_u8(value: u8) -> Self {
        match value & 0x0F {
            0 => Self::Regular,
            1 => Self::Directory,
            2 => Self::Symlink,
            3 => Self::BlockDevice,
            4 => Self::CharDevice,
            5 => Self::Fifo,
            6 => Self::Socket,
            _ => Self::Unknown,
        }
    }
}

/// Compression algorithms
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CompressionAlgo {
    None = 0,
    Lz4 = 1,
    Zstd = 2,
}

impl CompressionAlgo {
    pub fn from_u8(value: u8) -> Self {
        match value & 0x0F {
            1 => Self::Lz4,
            2 => Self::Zstd,
            _ => Self::None,
        }
    }
}

/// Filesystem errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FsError {
    NotFound,
    PermissionDenied,
    AlreadyExists,
    NotADirectory,
    IsADirectory,
    DirectoryNotEmpty,
    InvalidPath,
    OutOfSpace,
    TooManyFiles,
    IoError(String),
}

impl std::fmt::Display for FsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => write!(f, "File or directory not found"),
            Self::PermissionDenied => write!(f, "Permission denied"),
            Self::AlreadyExists => write!(f, "File already exists"),
            Self::NotADirectory => write!(f, "Not a directory"),
            Self::IsADirectory => write!(f, "Is a directory"),
            Self::DirectoryNotEmpty => write!(f, "Directory not empty"),
            Self::InvalidPath => write!(f, "Invalid path"),
            Self::OutOfSpace => write!(f, "No space left on device"),
            Self::TooManyFiles => write!(f, "Too many files"),
            Self::IoError(msg) => write!(f, "I/O error: {}", msg),
        }
    }
}

impl std::error::Error for FsError {}

pub type Result<T> = std::result::Result<T, FsError>;
