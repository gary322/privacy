// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

// UVCC_REQ_GROUP: uvcc_group_469c18d206813c25,uvcc_group_014ea1c86d0fe430

import {Test} from "forge-std/Test.sol";

import {AVLStakingManager} from "../src/AVLStakingManager.sol";
import {MockAVL} from "../src/MockAVL.sol";
import {ProviderBondRegistry} from "../src/ProviderBondRegistry.sol";
import {UVCCJobLedger} from "../src/UVCCJobLedger.sol";

contract UVCCJobLedgerTest is Test {
    MockAVL internal avl;
    AVLStakingManager internal staking;
    ProviderBondRegistry internal bonds;
    UVCCJobLedger internal ledger;

    uint256 internal pk0 = 0xA11CE;
    uint256 internal pk1 = 0xB0B;
    uint256 internal pk2 = 0xC0FFEE;
    address internal p0;
    address internal p1;
    address internal p2;
    address internal client;

    function setUp() public {
        p0 = vm.addr(pk0);
        p1 = vm.addr(pk1);
        p2 = vm.addr(pk2);
        client = vm.addr(0xD00D);

        avl = new MockAVL();
        staking = new AVLStakingManager(avl);
        bonds = new ProviderBondRegistry(avl, address(0));
        ledger = new UVCCJobLedger(staking, bonds);
        bonds.setJobLedger(address(ledger));
    }

    function _sig(uint256 pk, bytes32 digest) internal returns (bytes memory) {
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(pk, digest);
        return abi.encodePacked(r, s, v);
    }

    function test_createJob_and_submitFinal() public {
        bytes32 policyHash = keccak256("policy");
        bytes32 nonce = bytes32(uint256(123));
        address[3] memory parties = [p0, p1, p2];

        bytes32 jobId = ledger.computeJobId(policyHash, nonce);
        UVCCJobLedger.PolicyCommit memory pc = UVCCJobLedger.PolicyCommit({
            jobId: jobId,
            policyHash: policyHash,
            sidHash: keccak256("sid"),
            sgirHash: keccak256("sgir"),
            runtimeHash: keccak256("runtime"),
            fssDirHash: keccak256("fssdir"),
            preprocHash: keccak256("preproc"),
            backend: uint8(UVCCJobLedger.Backend.CRYPTO_CC_3PC),
            epoch: uint64(0)
        });
        bytes32 digestPc = ledger.hashPolicyCommit(pc);
        bytes memory sig0pc = _sig(pk0, digestPc);
        bytes memory sig1pc = _sig(pk1, digestPc);
        bytes memory sig2pc = _sig(pk2, digestPc);
        vm.prank(client);
        jobId = ledger.createJob(pc, parties, sig0pc, sig1pc, sig2pc);
        assertEq(jobId, ledger.computeJobId(policyHash, nonce));

        bytes32 finalRoot = bytes32(uint256(0xBEEF));
        bytes32 resultHash = bytes32(uint256(0xCAFE));
        bytes32 digestFc = ledger.hashFinalCommit(jobId, policyHash, finalRoot, resultHash);

        bytes memory sig0 = _sig(pk0, digestFc);
        bytes memory sig1 = _sig(pk1, digestFc);
        bytes memory sig2 = _sig(pk2, digestFc);

        vm.prank(client);
        ledger.submitFinal(jobId, finalRoot, resultHash, sig0, sig1, sig2);

        (
            bytes32 ph,
            bytes32 sidHash,
            bytes32 sgirHash,
            bytes32 runtimeHash,
            bytes32 fssDirHash,
            bytes32 preprocHash,
            UVCCJobLedger.Backend backend,
            uint64 epoch,
            address creator,
            uint64 createdAt,
            bool finalized,
            bytes32 fr,
            bytes32 rh,
            uint64 finalizedAt,
            bool challenged,
            address challenger,
            uint64 challengedAt,
            bool resolved,
            bool slashed
        ) = ledger.jobs(jobId);
        assertEq(ph, policyHash);
        assertEq(sidHash, pc.sidHash);
        assertEq(sgirHash, pc.sgirHash);
        assertEq(runtimeHash, pc.runtimeHash);
        assertEq(fssDirHash, pc.fssDirHash);
        assertEq(preprocHash, pc.preprocHash);
        assertEq(uint8(backend), pc.backend);
        assertEq(epoch, pc.epoch);
        assertTrue(finalized);
        assertEq(fr, finalRoot);
        assertEq(rh, resultHash);
        assertEq(creator, client);
        address[3] memory parties2 = ledger.getParties(jobId);
        assertEq(parties2[0], p0);
        assertEq(parties2[1], p1);
        assertEq(parties2[2], p2);
        assertEq(uint256(createdAt) > 0, true);
        assertEq(uint256(finalizedAt) > 0, true);
        assertEq(challenged, false);
        assertEq(challenger, address(0));
        assertEq(uint256(challengedAt), 0);
        assertEq(resolved, false);
        assertEq(slashed, false);
    }

    function test_submitFinal_rejects_bad_sig() public {
        bytes32 policyHash = keccak256("policy");
        bytes32 nonce = bytes32(uint256(123));
        address[3] memory parties = [p0, p1, p2];
        bytes32 jobId = ledger.computeJobId(policyHash, nonce);
        UVCCJobLedger.PolicyCommit memory pc = UVCCJobLedger.PolicyCommit({
            jobId: jobId,
            policyHash: policyHash,
            sidHash: keccak256("sid"),
            sgirHash: keccak256("sgir"),
            runtimeHash: keccak256("runtime"),
            fssDirHash: keccak256("fssdir"),
            preprocHash: keccak256("preproc"),
            backend: uint8(UVCCJobLedger.Backend.CRYPTO_CC_3PC),
            epoch: uint64(0)
        });
        bytes32 digestPc = ledger.hashPolicyCommit(pc);
        bytes memory sig0pc = _sig(pk0, digestPc);
        bytes memory sig1pc = _sig(pk1, digestPc);
        bytes memory sig2pc = _sig(pk2, digestPc);
        vm.prank(client);
        ledger.createJob(pc, parties, sig0pc, sig1pc, sig2pc);

        bytes32 finalRoot = bytes32(uint256(0xBEEF));
        bytes32 resultHash = bytes32(uint256(0xCAFE));
        bytes32 digestFc = ledger.hashFinalCommit(jobId, policyHash, finalRoot, resultHash);

        bytes memory sig0 = _sig(pk0, digestFc);
        bytes memory sig1 = _sig(pk1, digestFc);
        // Wrong signer for P2
        bytes memory sig2 = _sig(pk1, digestFc);

        vm.prank(client);
        vm.expectRevert("sig2 bad");
        ledger.submitFinal(jobId, finalRoot, resultHash, sig0, sig1, sig2);
    }
}


