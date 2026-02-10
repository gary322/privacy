// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {ERC20} from "openzeppelin-contracts/token/ERC20/ERC20.sol";

/// @notice Minimal ERC20 used for local demos/tests.
contract MockAVL is ERC20 {
    constructor() ERC20("AVL", "AVL") {}

    function mint(address to, uint256 amount) external {
        _mint(to, amount);
    }
}


