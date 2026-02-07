//! Register allocator for GPU DSL
//!
//! THE GPU IS THE COMPUTER.
//! Efficient register allocation = efficient GPU execution.
//!
//! Register layout:
//! - r0-r3:  Reserved (special/system)
//! - r4-r7:  Function arguments
//! - r8-r23: Temporaries (16 available)
//! - r24-r31: Callee-saved / spill area

use std::collections::HashMap;

/// Where a value lives
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Location {
    /// In a register
    Register(u8),
    /// Spilled to state memory at offset
    Spill(u32),
}

impl Location {
    pub fn reg(&self) -> Option<u8> {
        match self {
            Location::Register(r) => Some(*r),
            Location::Spill(_) => None,
        }
    }
}

/// Register allocator using linear scan
pub struct RegisterAllocator {
    /// Available temporary registers (r8-r23)
    free_temps: Vec<u8>,
    /// Variable name -> location
    var_locations: HashMap<String, Location>,
    /// Next spill slot offset
    next_spill: u32,
    /// Scratch register for temporaries (r30, r31)
    scratch_regs: Vec<u8>,
}

impl RegisterAllocator {
    pub fn new() -> Self {
        Self {
            // r8-r23 available as temporaries (16 regs)
            free_temps: (8..=23).rev().collect(),
            var_locations: HashMap::new(),
            next_spill: 0xF000, // High address for spill area
            scratch_regs: vec![31, 30], // r30, r31 for scratch
        }
    }

    /// Allocate a register for a variable
    pub fn allocate(&mut self, var: &str) -> Location {
        // Check if already allocated
        if let Some(loc) = self.var_locations.get(var) {
            return *loc;
        }

        // Try to get a free register
        let loc = if let Some(reg) = self.free_temps.pop() {
            Location::Register(reg)
        } else {
            // Spill to memory
            let slot = self.next_spill;
            self.next_spill += 1;
            Location::Spill(slot)
        };

        self.var_locations.insert(var.to_string(), loc);
        loc
    }

    /// Allocate a temporary register (for intermediate results)
    pub fn allocate_temp(&mut self) -> Location {
        if let Some(reg) = self.free_temps.pop() {
            Location::Register(reg)
        } else {
            // Use scratch register, will need to save/restore
            Location::Register(self.scratch_regs[0])
        }
    }

    /// Free a temporary register
    pub fn free_temp(&mut self, loc: Location) {
        if let Location::Register(reg) = loc {
            if reg >= 8 && reg <= 23 && !self.free_temps.contains(&reg) {
                self.free_temps.push(reg);
            }
        }
    }

    /// Get the location of a variable
    pub fn get(&self, var: &str) -> Option<Location> {
        self.var_locations.get(var).copied()
    }

    /// Free a variable's register
    pub fn free(&mut self, var: &str) {
        if let Some(Location::Register(reg)) = self.var_locations.remove(var) {
            if reg >= 8 && reg <= 23 {
                self.free_temps.push(reg);
            }
        }
    }

    /// Get a scratch register for temporary use
    pub fn get_scratch(&self, idx: usize) -> u8 {
        self.scratch_regs[idx % self.scratch_regs.len()]
    }

    /// Check if a register is available
    pub fn is_available(&self, reg: u8) -> bool {
        self.free_temps.contains(&reg)
    }
}

impl Default for RegisterAllocator {
    fn default() -> Self {
        Self::new()
    }
}
