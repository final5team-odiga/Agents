import React from "react";
import styled from "styled-components";

// 공통 이미지 컴포넌트
const StyledImage = styled.img`
  align-self: stretch;
`;

// 통일된 컬럼 프레임
const FlexColumn = styled.div`
  flex: 1 1 0;
  flex-direction: column;
  justify-content: flex-start;
  align-items: flex-start;
  gap: 12px;
  display: inline-flex;
`;

// 외곽 구조
const StyledFrame = styled.div`
  align-self: stretch;
  height: 800px;
  background: white;
  justify-content: flex-start;
  align-items: flex-start;
  gap: 40px;
  display: inline-flex;
`;

const StyledSection14 = styled.div`
  align-self: stretch;
  height: 800px;
  max-width: 1000px;
  padding: 114px 48px;
  flex-direction: column;
  justify-content: flex-start;
  align-items: center;
  display: inline-flex;
`;

export const Section14 = ({ imageUrl, subImageUrls }) => {
  return (
    <StyledSection14>
      <StyledFrame>
        {/* 왼쪽 큰 이미지 */}
        <FlexColumn
          style={{
            height: "647px",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <StyledImage src={imageUrl} style={{ height: "640px" }} />
        </FlexColumn>

        {/* 오른쪽 3개 이미지 */}
        <FlexColumn>
          <div style={{ display: "inline-flex", gap: 40, paddingBottom: 100 }}>
            <FlexColumn>
              <StyledImage src={subImageUrls[0]} style={{ height: "344px" }} />
            </FlexColumn>
            <FlexColumn>
              <StyledImage src={subImageUrls[1]} style={{ height: "344px" }} />
            </FlexColumn>
          </div>
          <div style={{ display: "inline-flex", gap: 40, paddingBottom: 100 }}>
            <FlexColumn>
              <StyledImage src={subImageUrls[2]} style={{ height: "257px" }} />
            </FlexColumn>
          </div>
        </FlexColumn>
      </StyledFrame>
    </StyledSection14>
  );
};
