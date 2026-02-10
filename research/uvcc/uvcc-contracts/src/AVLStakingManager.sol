// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

// UVCC_REQ_GROUP: uvcc_group_469c18d206813c25

import {IERC20} from "openzeppelin-contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "openzeppelin-contracts/token/ERC20/utils/SafeERC20.sol";

/// @notice Minimal staking manager (v1) — users lock AVL for job participation.
/// This is intentionally simple and deterministic for demo use.
contract AVLStakingManager {
    using SafeERC20 for IERC20;

    IERC20 public immutable avl;

    mapping(address => uint256) public stakeOf;

    event Staked(address indexed user, uint256 amount);
    event Unstaked(address indexed user, uint256 amount);

    constructor(IERC20 avl_) {
        avl = avl_;
    }

    function stake(uint256 amount) external {
        require(amount > 0, "amount=0");
        stakeOf[msg.sender] += amount;
        avl.safeTransferFrom(msg.sender, address(this), amount);
        emit Staked(msg.sender, amount);
    }

    function unstake(uint256 amount) external {
        require(amount > 0, "amount=0");
        uint256 bal = stakeOf[msg.sender];
        require(bal >= amount, "insufficient");
        stakeOf[msg.sender] = bal - amount;
        avl.safeTransfer(msg.sender, amount);
        emit Unstaked(msg.sender, amount);
    }
}


